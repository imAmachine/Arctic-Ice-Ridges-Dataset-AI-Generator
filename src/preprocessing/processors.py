import cv2
import numpy as np
import torch

from src.common.analyze_tools import FractalAnalyzer, FractalAnalyzerGPU
from src.common.interfaces import IProcessor
from src.common.utils import Utils

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
        return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel)


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
    
    def __init__(self, processor_name: str = None):
        super().__init__(processor_name)
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        points = self._find_farthest_points(image)
        angle = self._calculate_rotate_angle(points, image.shape)
        center = self._calculate_center(points)
        rotated_image = self._rotate_image(image, angle, center)
        
        self._result_value = angle
        
        return rotated_image
    
    def _find_farthest_points(self, image):
        """Находит 2 наиболее удалённые белые точки"""
        # Получаем координаты всех белых пикселей
        yx = np.column_stack(np.where(image > 0))  # (y, x) → для OpenCV меняем на (x, y) позже

        if len(yx) < 2:
            return None

        # Меняем порядок на (x, y) для OpenCV
        points = np.array([[x, y] for y, x in yx], dtype=np.int32)

        # Строим выпуклую оболочку
        hull = cv2.convexHull(points)

        # Используем rotating calipers для поиска самой дальней пары
        max_dist = 0
        farthest_pair = (tuple(hull[0][0]), tuple(hull[1][0]))

        n = len(hull)
        j = 1
        for i in range(n):
            while True:
                dist1 = np.linalg.norm(hull[i][0] - hull[j % n][0])
                dist2 = np.linalg.norm(hull[i][0] - hull[(j+1) % n][0])
                if dist2 > dist1:
                    j += 1
                else:
                    break
            if dist1 > max_dist:
                max_dist = dist1
                farthest_pair = (tuple(hull[i][0]), tuple(hull[j % n][0]))

        return farthest_pair
    
    def _calculate_rotate_angle(self, points, img_shape):
        """Вычисляет угол между линией (p1-p2) и диагональю изображения"""
        p1, p2 = points
        line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        h, w = img_shape
        diag_vec = np.array([w, -h])  # левая нижняя -> правая верхняя
        angle_rad = np.arctan2(line_vec[1], line_vec[0]) - np.arctan2(diag_vec[1], diag_vec[0])
        angle_deg = np.degrees(angle_rad)
        return (angle_deg + 90) % 180 - 90
    
    def _calculate_center(self, points):
        p1, p2 = points
        center = ((int(p1[0])+int(p2[0])) // 2, (int(p1[1])+int(p2[1])) // 2)
        return center
    
    def _rotate_image(self, img: np.ndarray, angle: float, center) -> np.ndarray:
        h, w = img.shape
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        _, binary = cv2.threshold(rotated_img, 70, 255, cv2.THRESH_BINARY)
        return binary


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