import random
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T


class OneOf(nn.Module):
    def __init__(self, transforms, p=1.0):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            t = random.choice(self.transforms)
            return t(x)
        return x

 
class RandomRotate(nn.Module):
    def __init__(self, angles=(90, 180, 270), p=0.8):
        super().__init__()
        self.angles = angles
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return T.functional.rotate(x, angle)
        return x


class BinarizeTransform(nn.Module):
    def __init__(self, threshold_type: Literal["mean_std", "bin", "byte"] = "bin"):
        super().__init__()
        self.threshold_type = threshold_type

    def forward(self, x: torch.Tensor):
        threshold = None
        
        if self.threshold_type == "mean_std":
            threshold = x.mean().item() + x.std().item()
        
        if self.threshold_type == "bin":
            threshold = 0.5
        
        if self.threshold_type == "byte":
            threshold = 255 / 2
        
        if threshold:
            return (x >= threshold).float()
        
        raise ValueError('Threshold value is None')


def get_common_transforms(target_img_size: int) -> T.Compose:
    max_crop = 1024
    return T.Compose([
        T.ToImage(),
        OneOf([T.RandomCrop((size, size)) for size in range(640, max_crop, 128)], p=1.0),
        RandomRotate(p=0.8),
        T.RandomHorizontalFlip(p=0.8),
        T.RandomVerticalFlip(p=0.8),
        T.Resize((target_img_size, target_img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToDtype(torch.float32, scale=True),
        BinarizeTransform(threshold_type="mean_std"),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_infer_transforms(target_img_size: int) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize((target_img_size, target_img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToDtype(torch.float32, scale=True),
        BinarizeTransform(threshold_type="mean_std"),
        T.Normalize(mean=[0.5], std=[0.5])
    ])