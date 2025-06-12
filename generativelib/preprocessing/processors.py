import cv2
import numpy as np
import torch

from generativelib.preprocessing.base import Processor


class Crop(Processor):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def process(self, image: np.ndarray) -> np.ndarray:
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


class AdjustToContent(Processor):
    def process(self, image: np.ndarray) -> np.ndarray:
        coords = np.argwhere(image > 0)
        if coords.size == 0:
            return image

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        return image[y_min:y_max, x_min:x_max]


class RotateMask(Processor):
    def __init__(self):
        super().__init__()
        self._angle = 0.0

    def process(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == bool:
            image = image.astype(np.uint8)

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
        rotation_mat = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
        rotation_mat = np.array(rotation_mat, dtype=np.float32)

        return cv2.warpAffine(# type: ignore
            image,
            rotation_mat,
            (int(w), int(h)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,)
        )


class TensorConverter(Processor):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def process(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).float().to(self.device)
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


class InferenceProcessor(Processor):
    def __init__(self, outpaint_ratio: float = 0.2):
        super().__init__()
        self.outpaint_ratio = outpaint_ratio
    
    def process(self, image: np.ndarray) -> torch.Tensor:
        processed_img = image.copy()
        h, w = image.shape[-2:]

        new_h = int(h * (1 + self.outpaint_ratio))
        new_w = int(w * (1 + self.outpaint_ratio))

        padded_image = np.zeros((new_h, new_w), dtype=processed_img.dtype)
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        padded_image[top:top + h, left:left + w] = processed_img
        
        return padded_image