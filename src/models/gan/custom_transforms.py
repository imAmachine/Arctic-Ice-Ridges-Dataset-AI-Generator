import random
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
