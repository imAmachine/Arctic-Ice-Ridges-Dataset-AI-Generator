import cv2
import numpy as np
import torch

from src.common.analyze_tools import FractalAnalyzer, FractalAnalyzerGPU
from ..common.interfaces import IProcessor
from ..common.image_processing import Utils

class CropProcessor(IProcessor):
    """Процессор для обрезки изображения"""
    
    def __init__(self, processor_name: str = None, crop_percent: int = 0):
        super().__init__(processor_name)
        self.crop_percent = crop_percent
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return Utils.crop_image(image, self.crop_percent)


class AutoAdjust(IProcessor):
    """Процессор для автоматического постраивания соотношения сторон изображения"""
    
    @property
    def PROCESSORS_NEEDED(self):
        return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        
        if h == w:
            return image
        elif h > w:
            diff = h - w
            top_crop = diff // 2
            bottom_crop = diff - top_crop
            return image[top_crop:h-bottom_crop, :]
        else:
            diff = w - h
            left_crop = diff // 2
            right_crop = diff - left_crop
            return image[:, left_crop:w-right_crop]


class Binarize(IProcessor):
    """Процессор для бинаризации изображения"""
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return Utils.binarize_by_threshold(image, threshold=0.1, max_val=1)


class Unbinarize(IProcessor):
    @property
    def PROCESSORS_NEEDED(self):
        return [Binarize]
    
    def process_image(self, image):
        return image * 255


class EnchanceProcessor(IProcessor):
    """Процессор для улучшения бинаризованного изображения с помощью морфологических операций"""
    
    def __init__(self, processor_name: str = None, kernel_size: int = 5):
        super().__init__(processor_name)
        self.kernel_size = kernel_size
    
    @property
    def PROCESSORS_NEEDED(self):
        return [Binarize]
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


class CropToContentProcessor(IProcessor):
    """Процессор для обрезки пустых краёв до границ контента."""

    def __init__(self, processor_name: str = None):
        super().__init__(processor_name)
    
    @property
    def PROCESSORS_NEEDED(self):
        return [Binarize]
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        return self._crop_to_content(image)

    def _crop_to_content(self, img: np.ndarray) -> np.ndarray:
        """
        Подгоняет размер изображения под область с контентом, удаляя пустые края.
        """
        coords = np.argwhere(img > 0)
        if coords.size == 0:
            return img
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        cropped = img[y_min:y_max, x_min:x_max]
        return cropped
    

class RotateMaskProcessor(IProcessor):
    """Процессор для выравнивания маски по максимальной диагонали выпуклой оболочки."""
    
    @property
    def PROCESSORS_NEEDED(self):
        return [Binarize, CropToContentProcessor]
    
    def __init__(self, processor_name: str = None):
        super().__init__(processor_name)
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        pass


class TensorConverterProcessor(IProcessor):
    """Процессор для преобразования numpy.ndarray в torch.Tensor"""
    
    def __init__(self, processor_name: str = None, device: str = "cpu"):
        super().__init__(processor_name)
        self.device = device
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            return image
            
        tensor_image = torch.from_numpy(image).float().to(self.device)
        
        if len(tensor_image.shape) == 2:
            tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image


class FractalDimensionProcessorGPU(IProcessor):
    """Процессор для расчета фрактальной размерности с использованием GPU"""
    @property
    def PROCESSORS_NEEDED(self):
        return [TensorConverterProcessor]
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        fr_dim = FractalAnalyzerGPU.calculate_fractal_dimension(
            *FractalAnalyzerGPU.box_counting(image), 
            device=image.device)
        
        self._result_value = fr_dim.item()
        
        return image


class FractalDimensionProcessorCPU(IProcessor):
    """Процессор для расчета фрактальной размерности с использованием CPU"""
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        fr_dim = FractalAnalyzer.calculate_fractal_dimension(
            *FractalAnalyzer.box_counting(image))
        
        self._result_value = fr_dim.item()
        
        return image