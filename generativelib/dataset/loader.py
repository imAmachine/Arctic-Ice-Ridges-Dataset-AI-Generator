
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import cv2
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2

from generativelib.dataset.base import BaseMaskProcessor
from generativelib.common.utils import Utils
from generativelib.model.enums import ExecPhase


class DatasetMaskingProcessor:
    def __init__(self, processors: List[BaseMaskProcessor]):
        self.processors = processors

    def create_mask(self, image: torch.Tensor):
        _, h, w = image.shape
        mask = torch.zeros((h, w), dtype=torch.float32, device=image.device, requires_grad=False)

        for processor in self.processors:
            mask = processor(mask)
        
        return mask

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor, is_inversed: bool=False) -> torch.Tensor:
        return image * ((1 - mask) if not is_inversed else mask)


class IceRidgeDataset(Dataset):
    def __init__(self, 
                 metadata: Dict[str, Dict], 
                 masking_processor: DatasetMaskingProcessor, 
                 model_transforms: torchvision.transforms.v2.Compose,
                 augmentations_per_image: int = 1,
                 is_trg_masked: bool = True):
        self.metadata = metadata
        self.masking_processor = masking_processor
        self.augmentations_per_image = augmentations_per_image
        self.model_transforms: torchvision.transforms.v2.Compose = model_transforms
        self.image_keys = list(metadata.keys())
        self.is_trg_masked = is_trg_masked

    def __len__(self) -> int:
        return len(self.image_keys) * self.augmentations_per_image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = (idx // self.augmentations_per_image)
        key = self.image_keys[img_idx]
        orig_path = self.metadata[key]['path']
        orig_img = Utils.cv2_load_image(orig_path, cv2.IMREAD_GRAYSCALE)
        
        transformed = self.model_transforms(orig_img)
        
        return self._process_img(transformed)
    
    def _process_img(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.masking_processor.create_mask(image)
        
        inp = self.masking_processor.apply_mask(image, mask)
        trg = image.clone()
        
        if self.is_trg_masked:
            trg = self.masking_processor.apply_mask(trg, mask, True)
        
        return inp, trg
<<<<<<< HEAD
=======
    
    def _get_processed_batch(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        return self._process_img(image)
>>>>>>> 433eae1 (add load_weights)
 

class DatasetCreator:
    def __init__(
        self, 
        metadata: Dict, 
        mask_processors: List[BaseMaskProcessor], 
        transforms: torchvision.transforms.v2.Compose, 
        dataset_params: Dict
    ):
        self.metadata = metadata
        self.mask_processor = DatasetMaskingProcessor(mask_processors)
        self.transforms = transforms
        self.dataset_params = dataset_params
    
    def create_loader(self, metadata: Optional[Dict]) -> DataLoader:
        if metadata is not None:
            dataset = IceRidgeDataset(
                metadata=metadata,
                masking_processor=self.mask_processor,
                model_transforms=self.transforms,
                augmentations_per_image=self.dataset_params.get("augs", 1),
                is_trg_masked=self.dataset_params.get("is_trg_masked", False)
            )
            
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.dataset_params.get("batch_size", 3),
                shuffle=self.dataset_params.get("shuffle", True),
                num_workers=self.dataset_params.get("workers", 4)
            )
            
            return loader
       
        raise ValueError('Metadata is empty while creating Dataloader')
    
    def create_loaders(self) -> Dict[ExecPhase, DataLoader]:        
        splitted = DatasetCreator.split_dataset_legacy(
            self.metadata, 
            valid_size_p=self.dataset_params.get("validation_size", 0.2)
        )
        
        train_metadata, valid_metadata = splitted.get(ExecPhase.TRAIN.name.lower()), splitted.get(ExecPhase.VALID.name.lower())
        
        # print(f"Размеры датасета: обучающий – {len(train_metadata)}; валидационный – {len(valid_metadata)}")
        
        return {
            ExecPhase.TRAIN: self.create_loader(train_metadata), 
            ExecPhase.VALID: self.create_loader(valid_metadata)
        }

    @staticmethod
    def split_dataset_legacy(metadata: Dict[str, Dict], valid_size_p: float) -> Dict[str, Dict[str, Dict[Any, Any]] | None]:
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
        
        return {
            ExecPhase.TRAIN.name.lower(): train_metadata if train_metadata else None,
            ExecPhase.VALID.name.lower(): val_metadata if val_metadata else None,
        }
