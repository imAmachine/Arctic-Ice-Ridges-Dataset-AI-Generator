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
from src.datasets.processors import MaskingProcessor

from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.preprocessing.processors import *

class IceRidgeDataset(Dataset):
    def __init__(self, metadata: Dict, dataset_processor: MaskingProcessor = None, with_target=False, transforms=None):
        self.processor = dataset_processor
        self.with_target = with_target
        self.metadata = metadata
        self.transforms = transforms
        self.image_keys = list(metadata.keys())
    
    def __len__(self) -> int:
        return len(self.image_keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает тройку (input, target, damage_mask) для индекса idx"""
        
        image = self._read_bin_image(self.image_keys[idx])  # чтение итерируемой картинки
        damaged, mask = self._get_processed_pair(input_img=image, masked=True)
        
        triplet = (damaged, image, mask)
        transformed_tensors = self.apply_transforms(triplet)
        
        binarized_tensors = [(im > 0.1).float() for im in transformed_tensors] # возврат к бинарному формату после трансформаций
        
        return binarized_tensors
    
    def _read_bin_image(self, metadata_key) -> np.ndarray:
        orig_meta = self.metadata[metadata_key]
        orig_path = orig_meta.get('path')
        img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        bin_img = Utils.binarize_by_threshold(img)
        return bin_img.astype(np.float32)
    
    def _get_processed_pair(self, input_img, masked):
        return self.processor.process(input_img.astype(np.float32), masked)
    
    def apply_transforms(self, objects: List):
        return [self.transforms(obj) for obj in objects]
        
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
        
        # 1. Собираем уникальные orig_path и группируем по ним файлы
        orig_to_files = {}
        for filename, info in metadata.items():
            orig_path = info.get('orig_path')
            if not orig_path:
                raise ValueError(f"Отсутствует orig_path для файла {filename}")
            
            if orig_path not in orig_to_files:
                orig_to_files[orig_path] = []
            orig_to_files[orig_path].append(filename)
        
        unique_origins = list(orig_to_files.keys())
        
        if len(unique_origins) < 2:
            raise ValueError(f"Недостаточно оригинальных изображений ({len(unique_origins)}). Нужно минимум 2")
        
        # 2. Случайно перемешиваем оригиналы
        random.shuffle(unique_origins)
        
        # 3. Определяем размер валидационной выборки (минимум 1 оригинал)
        val_size = max(1, int(len(unique_origins) * val_ratio))
        val_origins = unique_origins[:val_size]
        train_origins = unique_origins[val_size:]
        
        # 4. Формируем итоговые выборки
        train_metadata = {}
        val_metadata = {}
        
        for orig in train_origins:
            for filename in orig_to_files[orig]:
                train_metadata[filename] = metadata[filename]
        
        for orig in val_origins:
            for filename in orig_to_files[orig]:
                val_metadata[filename] = metadata[filename]
        
        # 5. Валидация и логирование
        if not val_metadata:
            raise RuntimeError("Валидационный набор пуст после разделения")
        
        print(
            f"Разделение данных: {len(train_metadata)} обучающих ({len(train_origins)} оригиналов), "
            f"{len(val_metadata)} валидационных ({len(val_origins)} оригиналов)"
        )
        
        return train_metadata, val_metadata


class IceRidgeDatasetGenerator:
    def __init__(self, augmentations_pipeline):
        self.augmentation_pipeline = augmentations_pipeline

    def generate(self, output_path, metadata, augmentations_per_image=5):
        """Генерирует аугментированные изображения и их метаданные"""
        if not self.augmentation_pipeline:
            return None
            
        os.makedirs(output_path, exist_ok=True)
        results_metadata = {}
        
        for file_name, file_metadata in metadata.items():
            aug_metadata = self._augment_image(
                file_metadata, 
                output_path, 
                augmentations_per_image
            )
            results_metadata.update(aug_metadata)
            
        return results_metadata
   
    def _augment_image(self, file_metadata, output_path, count):
        """Создает аугментированные версии одного изображения"""
        image_path = file_metadata.get('path')
        
        if not os.path.exists(image_path):
            return {}
            
        image = Utils.cv2_load_image(image_path)
        
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        new_metadata = {}
        
        # Генерация аугментаций
        for i in range(count):
            augmented_image = self.augmentation_pipeline(image=image)['image']
            binarized = Utils.binarize_by_threshold(augmented_image, 0.1, 1)
            
            new_filename = f"{base_name}_aug{i+1}{ext}"
            output_file_path = os.path.join(output_path, new_filename)
            
            # Сохранение изображения и метаданных
            cv2.imwrite(output_file_path, binarized * 255)
            new_metadata[f"{base_name}_aug{i+1}"] = {
                'path': output_file_path,
                'orig_path': file_metadata.get('path'),
                'fractal_dim': file_metadata.get('fractal_dimension')
            }
            
        print(f'Сгенерировано {count} аугментаций для {base_name}')
        return new_metadata


class DatasetCreator:
    def __init__(self, generated_path, original_data_path, preprocessed_data_path, images_extentions, 
                 model_transforms, preprocessors: List[IProcessor], augmentations_pipeline: A.Compose,
                 device):
        # === Init ===
        self.preprocessor = IceRidgeDatasetPreprocessor(preprocessors)
        self.dataset_generator = IceRidgeDatasetGenerator(augmentations_pipeline)
        self.dataset_processor = MaskingProcessor(mask_padding=0.20)
        self.device = device
        self.input_data_path = original_data_path
        
        # === Output paths ===
        self.generated_path = generated_path
        self.preprocessed_data_path = preprocessed_data_path
        self.preprocessed_metadata_json_path = os.path.join(self.preprocessed_data_path, 'metadata.json')
        self.generated_metadata_json_path = os.path.join(self.generated_path, 'metadata_generated.json')
        
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
   
    def augmentate_data(self, augmentations_per_image=5):
        if self.dataset_generator.augmentation_pipeline is not None or len(self.dataset_generator.augmentation_pipeline) > 0:
            os.makedirs(self.generated_path, exist_ok=True)
            
            preprocessed_metadata = self.from_json(self.preprocessed_metadata_json_path)
            generated_metadata = self.dataset_generator.generate(self.generated_path, preprocessed_metadata, augmentations_per_image)
            
            if generated_metadata is None:
                print('augmentation pipline is not defined! Add augmentation processors to dataset generator.')
                return
            
            self.to_json(generated_metadata, self.generated_metadata_json_path)
        else:
            print('Пайплайн аугментаций не объявлен!')
    
    def create_dataloaders(self, batch_size, shuffle, workers, val_ratio=0.2):
        dataset_metadata = self.from_json(self.generated_metadata_json_path)
        
        train_metadata, val_metadata = IceRidgeDataset.split_dataset(dataset_metadata, val_ratio=val_ratio)
        
        print(len(train_metadata), len(val_metadata))
        
        train_dataset = IceRidgeDataset(train_metadata, 
                                        dataset_processor=self.dataset_processor, 
                                        with_target=False,
                                        transforms=self.model_transforms)
        val_dataset = IceRidgeDataset(val_metadata, 
                                      dataset_processor=self.dataset_processor, 
                                      with_target=False,
                                      transforms=self.model_transforms)
        
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