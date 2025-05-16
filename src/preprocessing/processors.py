import cv2
import numpy as np
import torch

from src.common.interfaces import IProcessor
from src.common.utils import Utils

class Crop(IProcessor):
    """Процессор для обрезки изображения"""
    
    def __init__(self, processor_name: str = None, k: float = 1.0):
        super().__init__(processor_name)
        self.k = k
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        mask = image > 0
        if not mask.any():
            return image

        # белые пиксели по строкам и столбцам
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)

        # пороговые значения
        thr_r = row_counts.mean() - self.k * row_counts.std()
        thr_c = col_counts.mean() - self.k * col_counts.std()

        # индексы «достаточно плотных» строк и столбцов
        rows = np.where(row_counts >= thr_r)[0]
        cols = np.where(col_counts >= thr_c)[0]

        if rows.size == 0 or cols.size == 0:
            return image

        return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

# 
# Нужно доработать
# 

# class AutoAdjust(IProcessor):
#     """Процессор для автоматического ресайза изображения с сохранением соотношения сторон."""
    
#     def __init__(self, processor_name: str = None, target_size: int = 2048):
#         super().__init__(processor_name)
#         self.target_size = target_size
    
#     def process_image(self, image: np.ndarray) -> np.ndarray:
#         h, w = image.shape[:2]
#         if h == 0 or w == 0:
#             return image
        
#         scale = self.target_size / max(h, w)
#         new_w = int(w * scale)
#         new_h = int(h * scale)
        
#         interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        
#         resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
#         return resized

# class Binarize(IProcessor):
#     def process_image(self, image: np.ndarray) -> np.ndarray:
#         return Utils.binarize_by_threshold(image, threshold=image.std(), max_val=1)


# class Unbinarize(IProcessor):
#     def process_image(self, image):
#         return image * 255


# class EnchanceProcessor(IProcessor):
#     """Процессор для улучшения бинаризованного изображения с помощью морфологических операций"""
    
#     @property
#     def PROCESSORS_NEEDED(self):
#         return [Binarize]
    
#     def __init__(self, processor_name: str = None, kernel_size: int = 5):
#         super().__init__(processor_name)
#         self.kernel_size = kernel_size
    
#     def process_image(self, image: np.ndarray) -> np.ndarray:
#         # Убедимся, что тип правильный
#         binary_image = image.copy()
#         if image.dtype != np.uint8:
#             binary_image = image.astype(np.uint8)
        
#         # Структурные элементы по направлениям
#         vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.kernel_size))
#         horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, 1))
#         diag1 = np.eye(self.kernel_size, dtype=np.uint8)
#         diag2 = np.fliplr(diag1)

#         # Применяем морфологическое закрытие по 4 направлениям
#         repaired = binary_image.copy()
#         for kernel in [vertical, horizontal, diag1, diag2]:
#             closed = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, kernel)
#             repaired = cv2.bitwise_or(repaired, closed)  # объединяем результат

#         return repaired


class AdjustToContent(IProcessor):
    """Процессор для обрезки пустых краёв до границ контента."""

    def __init__(self, processor_name: str = None):
        super().__init__(processor_name)
    
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
    

class RotateMask(IProcessor):
    """Процессор, выравнивающий главную ось бинарной маски по диагонали изображения."""
    
    def __init__(self, processor_name: str = None):
        super().__init__(processor_name)
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        ys, xs = np.where(image > 0)
        if len(xs) < 2:
            self._result_value = 0.0
            return image
        
        pts = np.column_stack((xs, ys)).astype(np.int32)
        
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w_rect, h_rect), rect_angle = rect
        
        if w_rect < h_rect:
            angle = rect_angle + 90
        else:
            angle = rect_angle
        
        center = (cx, cy)
        
        h, w = image.shape
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        self._result_value = angle
        return rotated


class TensorConverter(IProcessor):
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
