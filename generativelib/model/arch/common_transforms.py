import random
from typing import List
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
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return (x >= x.std()).float()
        return x

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
        BinarizeTransform(p = 1.0),
    ])

def get_infer_transforms(target_img_size: int) -> List[T.Transform]:
    return T.Compose([
        T.ToImage(),
        T.Resize((target_img_size, target_img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToDtype(torch.float32, scale=True),
        BinarizeTransform(p = 1.0),
    ])