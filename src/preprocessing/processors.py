import cv2
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

class IProcessor(ABC):
    """Интерфейс процессора изображения."""

    dependencies: List[Type["IProcessor"]] = []

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, image: Any, metadata: Dict[str, Any]) -> Any:
        pass

    def get_metadata_value(self) -> str:
        return "True"


class Crop(IProcessor):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def process(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        mask = image > 0
        if not mask.any():
            return image

        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        thr_r = row_counts.mean() - self.k * row_counts.std()
        thr_c = col_counts.mean() - self.k * col_counts.std()
        rows = np.where(row_counts >= thr_r)[0]
        cols = np.where(col_counts >= thr_c)[0]

        if rows.size == 0 or cols.size == 0:
            return image

        return image[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]


class AdjustToContent(IProcessor):
    def process(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        coords = np.argwhere(image > 0)
        if coords.size == 0:
            return image

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        return image[y_min:y_max, x_min:x_max]


class RotateMask(IProcessor):
    def __init__(self):
        super().__init__()
        self._angle = 0.0

    def process(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        ys, xs = np.where(image > 0)
        if len(xs) < 2:
            self._angle = 0.0
            return image

        pts = np.column_stack((xs, ys)).astype(np.int32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w_rect, h_rect), rect_angle = rect

        angle = rect_angle + 90 if w_rect < h_rect else rect_angle
        self._angle = angle

        h, w = image.shape
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    def get_metadata_value(self) -> str:
        return str(round(self._angle, 2))


class TensorConverter(IProcessor):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

    def process(self, image: np.ndarray, metadata: Dict[str, Any]) -> torch.Tensor:
        tensor = torch.from_numpy(image).float().to(self.device)
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor
