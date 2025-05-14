import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, jaccard_score

class PrecisionMetric:
    """
    Метрика Precision:
     TP/(TP+FN)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, generated: np.ndarray, real: np.ndarray) -> float:
        return precision_score(real, generated, zero_division=1)

class F1Metric:
    def __init__(self):
        super().__init__()
    
    def __call__(self, generated: np.ndarray, real: np.ndarray) -> float:
        return f1_score(real, generated, zero_division=1)

class IoUMetric:
    def __init__(self):
        super().__init__()
    
    def __call__(self, generated: np.ndarray, real: np.ndarray) -> float:
        return jaccard_score(real, generated, zero_division=1)