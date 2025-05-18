import math

from dataclasses import dataclass
from typing import Callable
from torch.autograd import grad
from sklearn.metrics import f1_score, jaccard_score, precision_score
from src.common.analyze_tools import FractalAnalyzerGPU
import torch

@dataclass
class FractalMetric:
    def __call__(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
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

        return torch.tensor(fd_total / batch_size, device=fake_samples.device)

def sklearn_wrapper(fn, device, threshold: float = 0.5):
    def wrapper(gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # снимем с графа и скопируем на CPU
        y_pred = gen.detach().cpu().numpy().ravel()
        y_true = real.detach().cpu().numpy().ravel()

        # порогуем и приводим к 0/1
        y_pred_bin = (y_pred > threshold).astype(int)
        y_true_bin = (y_true > threshold).astype(int)

        # считаем метрику, на всякий случай обрабатываем zero_division
        val = fn(y_true_bin, y_pred_bin, zero_division=0)
        return torch.tensor(val, device=device)
    return wrapper


@dataclass
class GradientPenalty:
    model: Callable

    def __call__(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        # α ~ U(0,1)
        shape = [real_samples.size(0)] + [1] * (real_samples.dim() - 1)
        alpha = torch.rand(shape, device=real_samples.device)
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates.requires_grad_(True)

        d_interpolates = self.model(interpolates)
        
        # градиенты по входу
        grads = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True
        )[0]
        
        # norm по каждому семплу
        grads = grads.view(grads.size(0), -1)
        grad_norm = grads.norm(2, dim=1)

        return torch.mean((grad_norm - 1) ** 2)


@dataclass
class WassersteinLoss:
    model: Callable

    def __call__(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        real_pred = self.model(real_samples)
        fake_pred = self.model(fake_samples)
        return -torch.mean(real_pred) + torch.mean(fake_pred)
        

@dataclass
class AdversarialLoss:
    model: Callable

    def __call__(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        return -torch.mean(self.model(fake_samples))