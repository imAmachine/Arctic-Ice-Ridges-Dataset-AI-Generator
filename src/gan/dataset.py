import os
import random
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

import cv2
import numpy as np

from src.common.utils import Utils
from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.preprocessing.processors import *


class MaskingProcessor:
    def __init__(self, mask_padding=0.15):
        self.shift_percent = mask_padding
    
    def create_center_mask(self, shape) -> np.ndarray:
        """Создаём маску, которая накрывает центральную область изображения"""
        h, w = shape
        bh, bw = int(h * (1 - self.shift_percent)), int(w * (1 - self.shift_percent))
        top = (h - bh) // 2
        left = (w - bw) // 2
        mask = np.ones(shape, dtype=np.float32)
        mask[top:top + bh, left:left + bw] = 0.0
        return mask
    
    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Применяет повреждения к изображению"""
        img_size = image.shape
        damaged = image.copy()
        
        damage_mask = self.create_center_mask(shape=img_size)
        damaged *= (1 - damage_mask) # применение маски
            
        return (damaged, damage_mask)


class InferenceMaskingProcessor:
    def __init__(self, outpaint_ratio: float = 0.2):
        self.outpaint_ratio = outpaint_ratio

    def create_outpaint_mask(self, original_shape, target_shape) -> np.ndarray:
        """Создаёт маску для области, которую нужно сгенерировать (края)"""
        mask = np.ones(target_shape, dtype=np.float32)

        h0, w0 = original_shape
        h1, w1 = target_shape

        top = (h1 - h0) // 2
        left = (w1 - w0) // 2

        mask[top:top + h0, left:left + w0] = 0.0
        return mask

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Расширяет изображение и создаёт маску генерации на краях
        """
        h, w = image.shape[:2]
        new_h = int(h * (1 + self.outpaint_ratio))
        new_w = int(w * (1 + self.outpaint_ratio))

        padded_image = np.zeros((new_h, new_w), dtype=image.dtype)
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        padded_image[top:top + h, left:left + w] = image

        outpaint_mask = self.create_outpaint_mask((h, w), (new_h, new_w))

        return (padded_image, outpaint_mask)


class IceRidgeDataset(Dataset):
    def __init__(self, metadata: Dict[str, Dict], 
                 dataset_processor: 'MaskingProcessor', 
                 augmentations: Optional[Callable] = None, 
                 augmentations_per_image: int = 1,
                 model_transforms: Optional[Callable] = None, 
                 random_select: bool = False):
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
        self.augmentations_per_image = augmentations_per_image
        self.model_transforms = model_transforms
        self.random_select = random_select  # Флаг случайного выбора изображения

    def __len__(self) -> int:
        return len(self.image_keys) * self.augmentations_per_image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = random.choice(self.image_keys) if self.random_select else self.image_keys[idx]
        orig_meta = self.metadata[key]
        orig_path = orig_meta.get('path')
        img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        
        return IceRidgeDataset.prepare_data(img, self.processor, self.augmentations, self.model_transforms)
    
    @staticmethod
    def apply_transforms(model_transforms: Optional[Callable], images: List[np.ndarray]) -> torch.Tensor:
        if model_transforms is not None:
            return  [model_transforms(img) for img in images]
        return images
    
    @staticmethod
    def prepare_data(img: np.ndarray, processor: 'MaskingProcessor', augmentations: Optional[Callable], model_transforms: Optional[Callable]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original = Utils.binarize_by_threshold(img).astype(np.float32)
        img_aug = augmentations(to_tensor(original)) if augmentations is not None else original
        img_aug_np = img_aug.squeeze(0).numpy() 
        damaged, damage_mask = processor.process(image=img_aug_np)
        
        batch = (damaged, img_aug, damage_mask)
        
        tensors = IceRidgeDataset.apply_transforms(model_transforms, batch)
        binarized = [(x > 0.0).float() for x in tensors]
        
        return tuple(binarized)
    
    @staticmethod
    def split_dataset_legacy(metadata: Dict[str, Dict], val_ratio: float) -> Dict[str, Optional[Dict[str, Dict]]]:
        """
        Разделяет метаданные на обучающую и валидационную выборки,
        так чтобы данные одного оригинального изображения не оказывались в обеих выборках.
        """
        if not metadata:
            raise ValueError("Передан пустой словарь метаданных")
        
        unique_origins = list(metadata.keys())
        random.shuffle(unique_origins)
        val_size = 0
        train_origins, val_origins = [], []
        
        if val_ratio > 0.0:
            val_size = max(1, int(len(unique_origins) * val_ratio))

        train_origins = unique_origins[val_size:]
        val_origins = unique_origins[:val_size]
        
        train_metadata = {orig: metadata[orig] for orig in train_origins}
        val_metadata = {orig: metadata[orig] for orig in val_origins}
        
        print(f"{len(train_origins)} обучающих, {len(val_origins)} валидационных данных")
        
        return {"train": train_metadata if len(train_origins) > 0 else None, "valid": val_metadata if len(val_origins) > 0 else None}


class DatasetCreator:
    def __init__(self, generated_path, 
                 original_data_path, 
                 preprocessed_data_path, 
                 images_extentions, 
                 model_transforms, 
                 preprocessors: List, 
                 augmentations,
                 augs_per_img,
                 device):
        # Инициализация
        self.preprocessor = IceRidgeDatasetPreprocessor(preprocessors)
        self.dataset_processor = MaskingProcessor(mask_padding=0.20)
        self.augmentations = augmentations
        self.augs_per_img = augs_per_img
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
            Utils.to_json(self.preprocessor.metadata, self.preprocessed_metadata_json_path)
        else:
            print('Пайплайн предобработки не объявлен!')
    
    def create_loader(self, metadata, batch_size, shuffle, workers, is_augmentate, random_select):
        loader = None
        
        if metadata is not None:
            train_augs = self.augmentations if is_augmentate else None
            train_dataset = IceRidgeDataset(metadata, 
                                            dataset_processor=self.dataset_processor, 
                                            augmentations=train_augs,
                                            augmentations_per_image=self.augs_per_img,
                                            model_transforms=self.model_transforms,
                                            random_select=random_select)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers
            )
            
            return loader
    
    def create_train_dataloaders(self, batch_size, shuffle, workers, random_select=True, val_ratio=0.2, train_augmentations=True, val_augmentations=False) -> Dict[Literal['train', 'valid'], Dict]:
        if not os.path.exists(self.preprocessed_metadata_json_path):
            self.preprocess_data()
        
        dataset_metadata = Utils.from_json(self.preprocessed_metadata_json_path)
        splitted = IceRidgeDataset.split_dataset_legacy(dataset_metadata, val_ratio=val_ratio)
        train_metadata, valid_metadata = splitted.get('train'), splitted.get('valid')
        
        print(f"Размеры датасета: обучающий – {len(train_metadata)}; валидационный – {len(valid_metadata)}")
        
        train_loader = self.create_loader(train_metadata, batch_size, shuffle, workers, train_augmentations, random_select)
        valid_loader = self.create_loader(valid_metadata, batch_size, shuffle, workers, train_augmentations, random_select)
        
        return {'train': train_loader, 'valid': valid_loader}
