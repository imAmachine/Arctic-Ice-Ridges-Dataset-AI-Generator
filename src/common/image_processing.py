import cv2
import numpy as np


class Utils:
    @staticmethod
    def binarize_by_threshold(image: np.ndarray, threshold=127, max_val=1):
        _, binary = cv2.threshold(image, threshold, max_val, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def cv2_load_image(filename: str, cv2_read_mode: int=None):
        if cv2_read_mode is None:
            img = cv2.imread(filename)
        else:
            img = cv2.imread(filename, cv2_read_mode)
        
        if img is None:
            raise ValueError(f"Failed to load image {filename}")
        
        return img
    
    @staticmethod
    def crop_image(img: np.ndarray, crop_percent: int = 0) -> np.ndarray:
        if crop_percent > 0:
            h, w = img.shape
            crop_size_w, crop_size_h = int(w * (crop_percent / 100)), int(h * (crop_percent / 100))
            return img[crop_size_w:w-crop_size_w, crop_size_h:h-crop_size_h]
        return img