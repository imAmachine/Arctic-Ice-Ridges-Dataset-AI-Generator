import math
from dataclasses import dataclass

from torch import Tensor, tensor, sigmoid
from generativelib.common.analyze_tools import FractalAnalyzerGPU

@dataclass
class FractalMetric:
    def __call__(self, fake_samples: Tensor, real_samples: Tensor) -> Tensor:
        fd_total = 0.0
        batch_size = min(fake_samples.shape[0], 4)

        for i in range(batch_size):
            fake_img = fake_samples[i].detach().squeeze()
            real_img = real_samples[i].detach().squeeze()

            fd_fake = FractalAnalyzerGPU.calculate_fractal_dimension(
                *FractalAnalyzerGPU.box_counting(fake_img)
            )
            fd_real = FractalAnalyzerGPU.calculate_fractal_dimension(
                *FractalAnalyzerGPU.box_counting(real_img)
            )

            fd_diff = abs(fd_fake - fd_real)
            fd_total += math.exp(fd_diff) - 1

        return tensor(fd_total / batch_size, device=fake_samples.device)


def sklearn_wrapper(fn, device, threshold: float = 0.5):
    def wrapper(gen: Tensor, real: Tensor) -> Tensor:
        y_pred = sigmoid(gen.detach()).cpu().numpy().ravel()
        y_true = real.detach().cpu().numpy().ravel()

        # порогуем и приводим к 0/1
        y_pred_bin = (y_pred > threshold).astype(int)
        y_true_bin = (y_true > threshold).astype(int)

        # считаем метрику, на всякий случай обрабатываем zero_division
        val = fn(y_true_bin, y_pred_bin, zero_division=0)
        return tensor(val, device=device)
    return wrapper

