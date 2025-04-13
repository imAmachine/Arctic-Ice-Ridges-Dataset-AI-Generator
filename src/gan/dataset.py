import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import numpy as np

import os
import cv2
import json
import random
from typing import Dict, List, Tuple

from src.common.image_processing import Utils

from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.preprocessing.processors import *


class MaskingProcessor:
    def __init__(self, mask_padding=0.15):
        self.shift_percent = mask_padding
    
    def create_center_mask(self, shape):
        """Создаём маску, которая накрывает центральную область изображения"""
        h, w = shape
        bh, bw = int(h * (1 - self.shift_percent)), int(w * (1 - self.shift_percent))
        top = (h - bh) // 2
        left = (w - bw) // 2
        mask = np.ones(shape, dtype=np.float32)
        mask[top:top + bh, left:left + bw] = 0.0
        return mask
    
    def process(self, image: np.ndarray, masked=False) -> tuple:
        """Применяет повреждения к изображению"""
        img_size = image.shape
        damaged = image.copy()
        
        damage_mask = self.create_center_mask(shape=img_size)
        
        if masked:
            damaged *= (1 - damage_mask)
            
        return damaged, damage_mask
    


class IceRidgeDataset(Dataset):
    def __init__(self, metadata: Dict, dataset_processor: MaskingProcessor = None,
                 augmentations=None, model_transforms=None):
        self.processor = dataset_processor
        self.metadata = metadata
        self.image_keys = list(metadata.keys())
        
        self.augmentations = augmentations
        self.model_transforms = model_transforms
    
    def __len__(self) -> int:
        return len(self.image_keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original = self._read_bin_image(self.image_keys[idx])

        # === Аугментации
        if self.augmentations:
            augmented = self.augmentations(image=original)

        damaged, mask = self._get_processed_pair(input_img=augmented['image'], masked=True)
        
        # === Преобразования в тензоры / подготовка под модель
        damaged = self.apply_model_transforms(damaged)
        original = self.apply_model_transforms(augmented['image'])
        mask = self.apply_model_transforms(mask)

        # === Бинаризация (если нужно)
        triplet = (damaged, original, mask)
        binarized = [(x > 0.1).float() for x in triplet]

        return tuple(binarized)
    
    def _read_bin_image(self, metadata_key) -> np.ndarray:
        orig_meta = self.metadata[metadata_key]
        orig_path = orig_meta.get('path')
        img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        bin_img = Utils.binarize_by_threshold(img)
        return bin_img.astype(np.float32)
    
    def _get_processed_pair(self, input_img, masked):
        return self.processor.process(input_img.astype(np.float32), masked)
    
    def apply_model_transforms(self, img: np.ndarray) -> torch.Tensor:
        if self.model_transforms:
            return self.model_transforms(img)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
    @staticmethod
    def split_dataset(metadata: Dict, val_ratio=0.2, seed=42) -> Tuple[Dict, Dict]:
        """Разделяет метаданные на обучающую и валидационную выборки,
        гарантируя что аугментации одного изображения не разделяются.
        
        Args:
            metadata: Словарь метаданных в формате {filename: {info}}
            val_ratio: Доля данных для валидации (0.0-1.0)
            seed: Seed для воспроизводимости
            
        Returns:
            Кортеж (train_metadata, val_metadata)
        """
        if not metadata:
            raise ValueError("Передан пустой словарь метаданных")
        
        random.seed(seed)
        
        unique_origins = list(metadata.keys())
        random.shuffle(unique_origins)
        
        # 3. Определяем размер валидационной выборки (минимум 1 оригинал)
        val_size = max(1, int(len(unique_origins) * val_ratio))
        val_origins = unique_origins[:val_size]
        train_origins = unique_origins[val_size:]
        
        # 4. Формируем итоговые выборки
        train_metadata = {}
        val_metadata = {}
        
        for orig in train_origins:
            train_metadata[orig] = metadata[orig]
        
        for orig in val_origins:
            val_metadata[orig] = metadata[orig]
        
        print(
            f"Разделение данных: {len(train_metadata)} обучающих, "
            f"{len(val_metadata)} валидационных"
        )
        
        return train_metadata, val_metadata


class DatasetCreator:
    def __init__(self, generated_path, original_data_path, preprocessed_data_path, images_extentions, 
                 model_transforms, preprocessors: List[IProcessor], augmentations: A.Compose,
                 device):
        # === Init ===
        self.preprocessor = IceRidgeDatasetPreprocessor(preprocessors)
        self.dataset_processor = MaskingProcessor(mask_padding=0.20)
        self.augmentations = augmentations
        self.device = device
        self.input_data_path = original_data_path
        
        # === Output paths ===
        self.generated_path = generated_path
        self.preprocessed_data_path = preprocessed_data_path
        self.preprocessed_metadata_json_path = os.path.join(self.preprocessed_data_path, 'metadata.json')
        
        # === Params ===
        self.images_extentions = images_extentions
        self.model_transforms = model_transforms
        
    def preprocess_data(self):
        if self.preprocessor.processors is not None or len(self.preprocessor.processors) > 0:
            os.makedirs(self.preprocessed_data_path, exist_ok=True)
            
            self.preprocessor.process_folder(self.input_data_path, self.preprocessed_data_path,  self.images_extentions)
            if len(self.preprocessor.metadata) == 0:
                print('Метаданные не были получены после предобработки. Файл создан не будет!')
                return
            
            self.to_json(self.preprocessor.metadata, self.preprocessed_metadata_json_path)
        else:
            print('Пайплайн предобработки не объявлен!')
    
    def create_dataloaders(self, batch_size, shuffle, workers, val_ratio=0.2):
        if not os.path.exists(self.preprocessed_metadata_json_path):
            self.preprocess_data()
        
        dataset_metadata = self.from_json(self.preprocessed_metadata_json_path)
        
        train_metadata, val_metadata = IceRidgeDataset.split_dataset(dataset_metadata, val_ratio=val_ratio)
        
        print(len(train_metadata), len(val_metadata))
        
        train_dataset = IceRidgeDataset(train_metadata, 
                                        dataset_processor=self.dataset_processor, 
                                        augmentations=self.augmentations,
                                        model_transforms=self.model_transforms)
        val_dataset = IceRidgeDataset(val_metadata, 
                                      dataset_processor=self.dataset_processor, 
                                      augmentations=self.augmentations,
                                      model_transforms=self.model_transforms)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers
        )
        
        return train_loader, val_loader

    def to_json(self, metadata, path):
        with open(path, 'w+', encoding='utf8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    def from_json(self, path):
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)