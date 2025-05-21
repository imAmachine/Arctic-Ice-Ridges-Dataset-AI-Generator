import random
from typing import Tuple
import numpy as np
import torch
from src.dataset.base import BaseProcessStrategy


class Padding(BaseProcessStrategy):
    def __init__(self, ratio: float):
        self.ratio = ratio
    
    def _realization(self, mask: torch.Tensor):
        h, w = mask.shape
        top   = int(h * self.ratio)
        left  = int(w * self.ratio)
        bh, bw = h - 2 * top, w - 2 * left
        mask.fill_(1.0)
        mask[top : top + bh, left : left + bw] = 0.0
        
        return mask


class EllipsoidPadding(BaseProcessStrategy):
    def __init__(self, ratio: float = .15):
        super().__init__()
        self.ratio = ratio

    def _sample_axes(self, h: int, w: int) -> Tuple[float, float]:
        scale = 1 - self.ratio
        a = (w * scale) / 2 * random.uniform(.9, 1.1)
        b = (h * scale) / 2 * random.uniform(.9, 1.1)
        return a, b

    def _realization(self, mask: torch.Tensor) -> torch.Tensor:
        h, w = mask.shape
        cy, cx = h / 2.0, w / 2.0
        a, b   = self._sample_axes(h, w)
        angle  = random.uniform(0, np.pi)

        ys = torch.linspace(0, h - 1, h, device=mask.device)
        xs = torch.linspace(0, w - 1, w, device=mask.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        x_shift = xx - cx
        y_shift = yy - cy
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        x_rot = x_shift * cos_t + y_shift * sin_t
        y_rot = -x_shift * sin_t + y_shift * cos_t

        inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0

        mask.fill_(1.0)
        mask[inside] = 0.0
        
        return mask


class RandomHoles(BaseProcessStrategy):
    def __init__(self, count: int, min_sz: int, max_sz: int):
        self.count = count
        self.min_sz = min_sz
        self.max_sz = max_sz
        
    def _realization(self, mask: torch.Tensor) -> None:
        ys, xs = torch.nonzero(mask == 0, as_tuple=True)
        
        if ys.numel() == 0:
            return
        
        top,  bottom = ys.min().item(), ys.max().item()
        left, right  = xs.min().item(), xs.max().item()
        bh, bw       = bottom - top + 1, right - left + 1

        for _ in range(self.count):
            h = random.randint(self.min_sz, min(self.max_sz, bh))
            w = random.randint(self.min_sz, min(self.max_sz, bw))
            y = random.randint(top,  top + bh - h)
            x = random.randint(left, left + bw - w)
            mask[y : y + h, x : x + w] = 1.0
        
        return mask