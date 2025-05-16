import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Type

class IProcessor(ABC):
    """Interface for user defined image processing classes
    
    Attributes:
        NAME: Name identifier for the processor
        PROCESSORS_NEEDED: List of processor classes that should be applied before this processor
    """
    
    def __init__(self, processor_name: str = None):
        self.NAME = processor_name if processor_name else self.__class__.__name__
        self.metadata = {}
        self.VALUE = "False"
        self._result_value = None
    
    @property
    def PROCESSORS_NEEDED(self) -> List[Type['IProcessor']]:
        """Должен быть переопределен в дочерних классах"""
        return []
    
    @abstractmethod
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Метод обработки изображения, реализуемый в дочерних классах
        
        Args:
            image (np.ndarray): Изображение для обработки
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        pass
    
    def get_metadata_value(self) -> str:
        """Возвращает значение для записи в метаданные
        
        По умолчанию возвращает значение _result_value если оно установлено,
        иначе возвращает "True"
        """
        if self._result_value is not None:
            return str(self._result_value)
        return "True"
    
    def check_conditions(self, metadata: Dict[str, str]) -> bool:
        """Проверяет, выполнены ли все необходимые условия для запуска процессора
        
        Args:
            metadata: Текущие метаданные процесса обработки
            
        Returns:
            bool: True, если все необходимые процессоры были выполнены успешно
        """
        for processor_class in self.PROCESSORS_NEEDED:
            processor_name = processor_class.__name__
            proc_val = metadata.get(processor_name)
            if processor_name not in metadata or proc_val in ("False", None):
                raise Exception(f'Need {processor_name} before {self.NAME}')
    
    def process(self, image: np.ndarray, metadata: Dict[str, str]) -> np.ndarray:
        """Выполняет процесс обработки изображения
        
        Args:
            image: Изображение для обработки
            metadata: Метаданные процесса обработки
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        self.metadata = metadata
        
        self.check_conditions(metadata)
        processed_image = self.process_image(image)
        self.VALUE = self.get_metadata_value()
    
        metadata[self.NAME] = self.VALUE        
        return processed_image


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
