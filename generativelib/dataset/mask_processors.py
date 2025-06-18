import random
from typing import Tuple
import numpy as np
import torch
from generativelib.dataset.base import MaskProcessor


class Padding(MaskProcessor):
    def __init__(self, ratio: float = .2):
        self.ratio = ratio
    
    def _realization(self, cloned_mask: torch.Tensor):
        h, w = cloned_mask.shape
        top   = int(h * self.ratio)
        left  = int(w * self.ratio)
        bh, bw = h - 2 * top, w - 2 * left
        cloned_mask.fill_(1.0)
        cloned_mask[top : top + bh, left : left + bw] = 0.0


class EllipsoidPadding(MaskProcessor):
    def __init__(self, ratio: float = .2):
        self.ratio = ratio

    def _sample_axes(self, h: int, w: int) -> Tuple[float, float]:
        scale = 1 - self.ratio
        a = (w * scale) / 2 * random.uniform(.9, 1.1)
        b = (h * scale) / 2 * random.uniform(.9, 1.1)
        return a, b

    def _realization(self, cloned_mask: torch.Tensor):
        h, w = cloned_mask.shape
        cy, cx = h / 2.0, w / 2.0
        a, b   = self._sample_axes(h, w)
        angle  = random.uniform(0, np.pi)

        ys = torch.linspace(0, h - 1, h, device=cloned_mask.device)
        xs = torch.linspace(0, w - 1, w, device=cloned_mask.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        x_shift = xx - cx
        y_shift = yy - cy
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        x_rot = x_shift * cos_t + y_shift * sin_t
        y_rot = -x_shift * sin_t + y_shift * cos_t

        inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0

        cloned_mask.fill_(1.0)
        cloned_mask[inside] = 0.0


class RandomWindow(MaskProcessor):
    def __init__(self, window_scale_size: float=0.5):
        self.window_scale_size = window_scale_size
        
    def _realization(self, cloned_mask):
        h, w = cloned_mask.shape
        
        cloned_mask.fill_(1.0)
        top, left, bh, bw = 0, 0, h, w
        window_val = 0.0
        
        wind_h, wind_w = int((self.window_scale_size) * h), int((self.window_scale_size) * w)
        
        y = random.randint(top, top + bh - wind_h)
        x = random.randint(left, left + bw - wind_w)

        cloned_mask[y:y + wind_h, x:x + wind_w] = window_val
    


class RandomHoles(MaskProcessor):
    def __init__(self,
                 count: int = 1,
                 min_sz: int = 30,
                 max_sz: int = 40):
        self.count = count
        self.min_sz = min_sz
        self.max_sz = max_sz

    def _realization(self, cloned_mask: torch.Tensor):
        ys, xs = torch.nonzero(cloned_mask == 0, as_tuple=True)
        
        if ys.numel() == 0:
            return
        
        top    = int(ys.min().item())
        bottom = int(ys.max().item())
        left   = int(xs.min().item())
        right  = int(xs.max().item())
        
        bh = bottom - top + 1
        bw = right  - left + 1
        hole_val = 1.0

        for _ in range(self.count):
            hole_h = random.randint(self.min_sz, min(self.max_sz, bh))
            hole_w = random.randint(self.min_sz, min(self.max_sz, bw))

            y = random.randint(top,  top + bh - hole_h)
            x = random.randint(left, left + bw - hole_w)

            cloned_mask[y:y + hole_h, x:x + hole_w] = hole_val


MASK_PROCESSORS = {
    Padding.__name__: Padding,
    EllipsoidPadding.__name__: EllipsoidPadding,
    RandomHoles.__name__: RandomHoles,
    RandomWindow.__name__: RandomWindow,
}
