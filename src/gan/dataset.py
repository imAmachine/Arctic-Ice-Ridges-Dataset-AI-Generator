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
    def __init__(self, metadata: Dict, dataset_processor, augmentations=None, model_transforms=None, random_select: bool = False):
        """
        Args:
            metadata: словарь с метаданными изображений
            dataset_processor: экземпляр класса для обработки изображений (например, MaskingProcessor)
            augmentations: экземпляр A.Compose с аугментациями
            model_transforms: преобразования для подготовки под модель (например, нормализация, to tensor)
            random_select: если True – при каждом __getitem__ выбирается случайное изображение из metadata,
                           что бывает полезно при очень маленьком датасете.
        """
        self.processor = dataset_processor
        self.metadata = metadata
        self.image_keys = list(metadata.keys())
        self.augmentations = augmentations
        self.model_transforms = model_transforms
        self.random_select = random_select  # Флаг случайного выбора изображения

    def __len__(self) -> int:
        return len(self.image_keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Если датасет очень маленький, можно случайно выбирать изображение, а не по порядку.
        if self.random_select:
            rand_idx = random.randint(0, len(self.image_keys) - 1)
            key = self.image_keys[rand_idx]
        else:
            key = self.image_keys[idx]
            
        original = self._read_bin_image(key)
    
        img = original
        if self.augmentations is not None:
            img = self.augmentations(image=original)['image']
    
        # Получаем пару: повреждённое изображение и маску (обработка внутри processor)
        damaged, mask = self._get_processed_pair(input_img=img, masked=True)
        
        # Преобразование изображений под формат модели
        damaged = self.apply_model_transforms(damaged)
        original_transformed = self.apply_model_transforms(img)
        mask_transformed = self.apply_model_transforms(mask)
    
        # Бинаризация, если необходимо (например, для маски и иных тензоров)
        triplet = (damaged, original_transformed, mask_transformed)
        binarized = [(x > 0.1).float() for x in triplet]
        
        return tuple(binarized)
    
    def _read_bin_image(self, metadata_key) -> np.ndarray:
        orig_meta = self.metadata[metadata_key]
        orig_path = orig_meta.get('path')
        img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        bin_img = Utils.binarize_by_threshold(img)
        return bin_img.astype(np.float32)
    
    def _get_processed_pair(self, input_img, masked: bool):
        return self.processor.process(input_img.astype(np.float32), masked)
    
    def apply_model_transforms(self, img: np.ndarray) -> torch.Tensor:
        if self.model_transforms:
            return self.model_transforms(img)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
    @staticmethod
    def split_dataset(metadata: Dict, val_ratio) -> Tuple[Dict, Dict]:
        """
        Разделяет метаданные на обучающую и валидационную выборки,
        так чтобы данные одного оригинального изображения не оказывались в обеих выборках.
        """
        if not metadata:
            raise ValueError("Передан пустой словарь метаданных")
        
        unique_origins = list(metadata.keys())
        random.shuffle(unique_origins)
        
        val_size = max(0, int(len(unique_origins) * val_ratio))
        val_origins = unique_origins[:val_size]
        train_origins = unique_origins[val_size:]
        
        train_metadata = {orig: metadata[orig] for orig in train_origins}
        val_metadata = {orig: metadata[orig] for orig in val_origins}
        
        print(f"Разделение данных: {len(train_metadata)} обучающих, {len(val_metadata)} валидационных")
        
        return train_metadata, val_metadata


class DatasetCreator:
    def __init__(self, generated_path, original_data_path, preprocessed_data_path, images_extentions, 
                 model_transforms, preprocessors: List, augmentations: A.Compose,
                 device):
        # Инициализация
        self.preprocessor = IceRidgeDatasetPreprocessor(preprocessors)
        self.dataset_processor = MaskingProcessor(mask_padding=0.20)
        self.augmentations = augmentations
        self.device = device
        self.input_data_path = original_data_path
        
        # Пути для выходных данных
        self.generated_path = generated_path
        self.preprocessed_data_path = preprocessed_data_path
        self.preprocessed_metadata_json_path = os.path.join(self.preprocessed_data_path, 'metadata.json')
        
        self.images_extentions = images_extentions
        self.model_transforms = model_transforms
        
    def preprocess_data(self):
        if self.preprocessor.processors is not None and len(self.preprocessor.processors) > 0:
            os.makedirs(self.preprocessed_data_path, exist_ok=True)
            self.preprocessor.process_folder(self.input_data_path, self.preprocessed_data_path, self.images_extentions)
            if len(self.preprocessor.metadata) == 0:
                print('Метаданные не были получены после предобработки. Файл создан не будет!')
                return
            self.to_json(self.preprocessor.metadata, self.preprocessed_metadata_json_path)
        else:
            print('Пайплайн предобработки не объявлен!')
    
    def create_dataloaders(self, batch_size, shuffle, workers, random_select, val_ratio=0.2, val_augmentations=False, train_augmentations=True):
        if not os.path.exists(self.preprocessed_metadata_json_path):
            self.preprocess_data()
        
        dataset_metadata = self.from_json(self.preprocessed_metadata_json_path)
        
        train_metadata, val_metadata = IceRidgeDataset.split_dataset(dataset_metadata, val_ratio=val_ratio)
        
        print(f"Размеры датасета: обучающий – {len(train_metadata)}; валидационный – {len(val_metadata)}")
        
        train_augs = self.augmentations if train_augmentations else None
        train_dataset = IceRidgeDataset(train_metadata, 
                                        dataset_processor=self.dataset_processor, 
                                        augmentations=train_augs,
                                        model_transforms=self.model_transforms,
                                        random_select=random_select)
        
        val_augs = self.augmentations if val_augmentations else None
        val_dataset = IceRidgeDataset(val_metadata, 
                                      dataset_processor=self.dataset_processor, 
                                      augmentations=val_augs,
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
