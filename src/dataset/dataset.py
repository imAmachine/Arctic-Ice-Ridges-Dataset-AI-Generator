
from collections import defaultdict
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import cv2
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np


from src.dataset.structs import MaskRegion, ProcessingStrategies
from src.common.enums import ExecPhase as phases
from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.common.utils import Utils


class DatasetMaskingProcessor:
    def __init__(self, mask_params: Dict, processing_strats: ProcessingStrategies):
        self.mask_params = mask_params
        self.processing_strats = processing_strats
    
    def _create_mask_region(self, height: int, width: int, device, dtype) -> MaskRegion:
        bh = int(height * (1 - self.mask_params['padding']))
        bw = int(width  * (1 - self.mask_params['padding']))
        top  = (height - bh) // 2
        left = (width  - bw) // 2

        mask = np.ones((height, width), device=device, dtype=dtype)
        mask[top:top + bh, left:left + bw] = 0.0

        return MaskRegion(mask, top, left, bh, bw)
    
    def process(self, image: np.ndarray) -> torch.Tensor:
        processed_img = image.copy()
        _, h, w = processed_img.shape
        
        region = self._create_mask_region(h, w, device=image.device, dtype=image.dtype)
        self.processing_strats.apply_all(region, self.mask_params['processors'])
        
        return processed_img * (1 - region.mask)    


class IceRidgeDataset(Dataset):
    def __init__(self, metadata: Dict[str, Dict], 
                 masking_processor: 'DatasetMaskingProcessor', 
                 augmentations_per_image: int = 1,
                 model_transforms: Optional[Callable] = None):
        self.masking_processor = masking_processor
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
        inp = self.masking_processor.process(transformed_trg)
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
    def __init__(
            self,
            input_preprocessor: IceRidgeDatasetPreprocessor,
            masking_processor: DatasetMaskingProcessor,
            model_transforms: 'torchvision.transforms.Compose', 
            augs_per_img: int = 1, 
        ):
        # Инициализация
        self.preprocessor = input_preprocessor # первоначальная предобработка исходных данных
        self.dataset_processor = masking_processor # Процессор для маскирования и создания датасета
        
        self.model_transforms = model_transforms # трансформации необходимые при передаче батча в модель
        self.augs_per_img = augs_per_img # количество аугментаций на снимок
    
    def create_loader(self, metadata, batch_size, shuffle, workers):
        loader = None
        
        if metadata is not None:
            train_dataset = IceRidgeDataset(
                metadata=metadata, 
                masking_processor=self.dataset_processor,
                augmentations_per_image=self.augs_per_img,
                model_transforms=self.model_transforms
            )
            
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers
            )
            
            return loader
    
    def get_dataloaders(self, batch_size, shuffle, workers, val_ratio=0.2) -> Dict[phases, Dict]:
        # предобработка входных снимков
        dataset_metadata = defaultdict()
        
        if not self.preprocessor.is_metadata_exist():
            self.preprocessor.process_folder()
            dataset_metadata = self.preprocessor.metadata
        else:
            dataset_metadata = Utils.from_json(self.preprocessor.metadata_json_path)
        
        splitted = IceRidgeDataset.split_dataset_legacy(dataset_metadata, val_ratio=val_ratio)
        train_metadata, valid_metadata = splitted.get(phases.TRAIN), splitted.get(phases.VALID)
        
        print(f"Размеры датасета: обучающий – {len(train_metadata)}; валидационный – {len(valid_metadata)}")
        
        train_loader = self.create_loader(train_metadata, batch_size, shuffle, workers)
        valid_loader = self.create_loader(valid_metadata, batch_size, shuffle, workers)
        
        return train_loader, valid_loader
