import random
from typing import Dict
import numpy as np
from src.dataset.structs import BaseProcessStrategy, MaskRegion


class RandomHoleStrategy(BaseProcessStrategy):
    def apply(self, region: MaskRegion, params: Dict):
        for _ in range(params['count']):
            hole_w = np.random.randint(params['min_size'], params['max_size'] + 1)
            hole_h = np.random.randint(params['min_size'], params['max_size'] + 1)
            
            y = random.randint(region.top, region.top + region.bh - hole_h - 1)
            x = random.randint(region.left, region.left + region.bw - hole_w - 1)

            region.mask[y:y+hole_h, x:x+hole_w] = 1.0