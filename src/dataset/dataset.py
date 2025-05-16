import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import torchvision
import cv2

from src.common.enums import ExecPhase as phases
from src.dataset.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.common.utils import Utils

@dataclass
class MaskRegion:
    mask: torch.Tensor
    top: int
    left: int
    bh: int
    bw: int

class MaskingProcessor:
    def __init__(self, masking_prm: Dict):
        self.masking_prm = masking_prm

    def create_border_mask(self, height: int, width: int, device, dtype) -> MaskRegion:
        bh = int(height * (1 - self.masking_prm['mask_padding']))
        bw = int(width  * (1 - self.masking_prm['mask_padding']))
        top  = (height - bh) // 2
        left = (width  - bw) // 2

        mask = np.ones((height, width), device=device, dtype=dtype)
        mask[top:top + bh, left:left + bw] = 0.0

        return MaskRegion(mask, top, left, bh, bw)
    
    def create_random_hole(self, region: MaskRegion) -> None:
        for _ in range(self.masking_prm['num_holes']):
            hole_w = np.random.randint(self.masking_prm['min_hole_size'], self.masking_prm['max_hole_size'] + 1)
            hole_h = np.random.randint(self.masking_prm['min_hole_size'], self.masking_prm['max_hole_size'] + 1)
            
            y = random.randint(region.top, region.top + region.bh - hole_h - 1)
            x = random.randint(region.left, region.left + region.bw - hole_w - 1)

            region.mask[y:y+hole_h, x:x+hole_w] = 1.0

    def process(self, image: np.ndarray) -> torch.Tensor:
        damaged = image.copy()
        _, h, w = damaged.shape
        region = self.create_border_mask(h, w, device=image.device, dtype=image.dtype)
        if self.masking_prm['num_holes'] > 0:
            self.create_random_hole(region)
        damaged = damaged * (1 - region.mask)

        return damaged


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
                 augmentations_per_image: int = 1,
                 model_transforms: Optional[Callable] = None):
        self.preprocessor = dataset_processor
        self.metadata = metadata
        self.image_keys = list(metadata.keys())
        self.augmentations_per_image = augmentations_per_image
        self.model_transforms: 'torchvision.transforms.Compose' = model_transforms

    def __len__(self) -> int:
        return len(self.image_keys) * self.augmentations_per_image
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_idx = (idx // self.augmentations_per_image)
        key = self.image_keys[img_idx]
        orig_path = self.metadata[key]['path']
        orig_img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        
        return self.get_processed_batch(orig_img)
    
    def process_img(self, img: np.ndarray) -> 'List[np.ndarray]':
        transformed_trg = self.model_transforms(img).numpy()
        inp = self.preprocessor.process(transformed_trg)
        return inp, transformed_trg
    
    def get_processed_batch(self, img: np.ndarray) -> Tuple[torch.Tensor,]:
        original = img.astype(np.float32)
        
        batch = self.process_img(original)
        inp, trg = [Utils.binarize_by_threshold(el, threshold=el.std(), max_val=1.0) for el in batch]

        return inp, trg
    
    #
    #   НЕОБХОДИМО ПЕРЕРАБОТАТЬ ЛОГИКУ РАЗДЕЛЕНИЯ ДАТАСЕТА
    #
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
        
        return {phases.TRAIN: train_metadata if len(train_origins) > 0 else None, phases.VALID: val_metadata if len(val_origins) > 0 else None}


class DatasetCreator:
    def __init__(self, device: torch.device,
                 output_path: str, 
                 original_images_path: str, 
                 preprocessed_images_path: str, 
                 images_ext: List[str], 
                 model_transforms: 'torchvision.transforms.Compose', 
                 preprocessors: List,
                 masking_prm: dict,
                 augs_per_img: int = 1, 
                 ):
        # Инициализация
        self.preprocessor = IceRidgeDatasetPreprocessor(preprocessors)
        self.dataset_processor = MaskingProcessor(masking_prm) # Процессор для обработки тренировочных изображений
        self.model_transforms = model_transforms # трансформации необходимые при передаче батча в модель
        self.augs_per_img = augs_per_img # количество аугментаций на снимок
        self.device = device
        self.input_data_path = original_images_path # путь к входным снимкам
        self.images_ext = images_ext # список расширений файлов
        
        # Пути для выходных данных
        self.generated_path = output_path
        self.preprocessed_data_path = preprocessed_images_path
        self.preprocessed_metadata_json_path = os.path.join(self.preprocessed_data_path, 'metadata.json')

    def preprocess_data(self):
        if self.preprocessor.processors is not None and len(self.preprocessor.processors) > 0:
            os.makedirs(self.preprocessed_data_path, exist_ok=True)
            self.preprocessor.process_folder(self.input_data_path, self.preprocessed_data_path, self.images_ext)
            if len(self.preprocessor.metadata) == 0:
                print('Метаданные не были получены после предобработки. Файл создан не будет!')
                return
            Utils.to_json(self.preprocessor.metadata, self.preprocessed_metadata_json_path)
        else:
            print('Пайплайн предобработки не объявлен!')
    
    def create_loader(self, metadata, batch_size, shuffle, workers):
        loader = None
        
        if metadata is not None:
            train_dataset = IceRidgeDataset(metadata, 
                                            dataset_processor=self.dataset_processor,
                                            augmentations_per_image=self.augs_per_img,
                                            model_transforms=self.model_transforms)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers
            )
            
            return loader
    
    def get_dataloaders(self, batch_size, shuffle, workers, val_ratio=0.2) -> Dict[phases, Dict]:
        if not os.path.exists(self.preprocessed_metadata_json_path):
            self.preprocess_data()
        
        dataset_metadata = Utils.from_json(self.preprocessed_metadata_json_path)
        splitted = IceRidgeDataset.split_dataset_legacy(dataset_metadata, val_ratio=val_ratio)
        train_metadata, valid_metadata = splitted.get(phases.TRAIN), splitted.get(phases.VALID)
        
        print(f"Размеры датасета: обучающий – {len(train_metadata)}; валидационный – {len(valid_metadata)}")
        
        train_loader = self.create_loader(train_metadata, batch_size, shuffle, workers)
        valid_loader = self.create_loader(valid_metadata, batch_size, shuffle, workers)
        
        return train_loader, valid_loader
