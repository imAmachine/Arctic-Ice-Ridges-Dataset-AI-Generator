import numpy as np


class ShiftProcessor:
    def __init__(self, shift_percent=0.15):
        self.shift_percent = shift_percent
    
    def create_center_mask(self, shape):
        """Создаём маску, которая накрывает центральную область изображения"""
        h, w = shape
        bh, bw = int(h * (1 - self.shift_percent)), int(w * (1 - self.shift_percent))
        top = (h - bh) // 2
        left = (w - bw) // 2
        mask = np.ones(shape, dtype=np.float32)
        mask[top:top + bh, left:left + bw] = 0.0
        return mask
    
    def process(self, image: np.ndarray, masked=False) -> tuple:
        """Применяет повреждения к изображению"""
        img_size = image.shape
        damaged = image.copy()
        
        damage_mask = self.create_center_mask(shape=img_size)
        
        if masked:
            damaged *= (1 - damage_mask)
            
        return damaged, damage_mask
    