
from collections import defaultdict
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import cv2
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np


from src.dataset.base import MaskRegion, ProcessingStrategies
from src.common.enums import ExecPhase
from src.preprocessing.preprocessor import DataPreprocessor
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
    def __init__(self, 
                 metadata: Dict[str, Dict], 
                 masking_processor: 'DatasetMaskingProcessor', 
                 augmentations_per_image: int = 1,
                 model_transforms: Optional[Callable] = None):
        self.metadata = metadata
        self.masking_processor = masking_processor
        self.augmentations_per_image = augmentations_per_image
        self.model_transforms: 'torchvision.transforms.Compose' = model_transforms
        self.image_keys = list(metadata.keys())

    def __len__(self) -> int:
        return len(self.image_keys) * self.augmentations_per_image
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_idx = (idx // self.augmentations_per_image)
        key = self.image_keys[img_idx]
        orig_path = self.metadata[key]['path']
        orig_img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        
        return self._get_processed_batch(orig_img)
    
    def _process_img(self, img: np.ndarray) -> 'List[np.ndarray]':
        transformed_trg = self.model_transforms(img).numpy()
        inp = self.masking_processor.process(transformed_trg)
        return inp, transformed_trg
    
    def _get_processed_batch(self, img: np.ndarray) -> Tuple[torch.Tensor,]:
        original = img.astype(np.float32)
        
        batch = self._process_img(original)
        inp, trg = [Utils.binarize_by_threshold(el, threshold=el.std(), max_val=1.0) for el in batch]

        return inp, trg
 

class DatasetCreator:
    def __init__(self, metadata, mask_processor, transforms, augs_per_img, valid_size_p, shuffle, batch_size, workers):
        self.metadata = metadata
        self.mask_processor = mask_processor
        self.transforms = transforms
        self.augs_per_img = augs_per_img
        self.valid_size_p = valid_size_p
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.workers = workers
    
    def create_loader(self, metadata):
        loader = None
        
        if metadata is not None:
            dataset = IceRidgeDataset(
                metadata=metadata,
                masking_processor=self.mask_processor,
                model_transforms=self.transforms,
                augmentations_per_image=self.augs_per_img
            )
            
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.workers
            )
            
        return loader
    
    def create_loaders(self) -> Dict[ExecPhase, Dict]:        
        splitted = DatasetCreator.split_dataset_legacy(self.metadata, valid_size_p=self.valid_size_p)
        train_metadata, valid_metadata = splitted.get(ExecPhase.TRAIN), splitted.get(ExecPhase.VALID)
        
        # print(f"Размеры датасета: обучающий – {len(train_metadata)}; валидационный – {len(valid_metadata)}")
        
        return {
            ExecPhase.TRAIN: self.create_loader(train_metadata), 
            ExecPhase.VALID: self.create_loader(valid_metadata)
        }

    @staticmethod
    def split_dataset_legacy(metadata: Dict[str, Dict], valid_size_p: float) -> Dict[str, Optional[Dict[str, Dict]]]:
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
        
        if valid_size_p > 0.0:
            val_size = max(1, int(len(unique_origins) * valid_size_p))

        train_origins = unique_origins[val_size:]
        val_origins = unique_origins[:val_size]
        
        train_metadata = {orig: metadata[orig] for orig in train_origins}
        val_metadata = {orig: metadata[orig] for orig in val_origins}
        
        print(f"{len(train_origins)} обучающих, {len(val_origins)} валидационных данных")
        
        return {ExecPhase.TRAIN: train_metadata if len(train_origins) > 0 else None, ExecPhase.VALID: val_metadata if len(val_origins) > 0 else None}