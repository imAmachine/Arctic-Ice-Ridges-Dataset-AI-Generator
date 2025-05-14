from torch.autograd import grad
from sklearn.metrics import f1_score, jaccard_score, precision_score
from src.common.analyze_tools import FractalAnalyzerGPU
import torch.nn as nn
import torch

def fractal_metric(generated: torch.Tensor, target: torch.Tensor) -> float:
    """
    Считает среднюю разницу фрактальной размерности между сгенерированным изображением и ground truth
    """
    fd_total = 0.0
    batch_size = min(generated.shape[0], 4)

    for i in range(batch_size):
        gen_img = generated[i].detach().squeeze()
        tgt_img = target[i].detach().squeeze()

        fd_gen = FractalAnalyzerGPU.calculate_fractal_dimension(
            *FractalAnalyzerGPU.box_counting(gen_img)
        )
        fd_target = FractalAnalyzerGPU.calculate_fractal_dimension(
            *FractalAnalyzerGPU.box_counting(tgt_img)
        )

        fd_total += abs(fd_gen - fd_target)

    return fd_total / batch_size

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


class GradientPenalty(nn.Module):
    """
    Вычисляет градиентный штраф WGAN-GP:
      E[(||∇ D(α·real + (1−α)·fake)||₂ − 1)²]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
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


class WassersteinLoss(nn.Module):
    """
    WGAN loss: −E[D(real)] + E[D(fake)]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        real_pred = self.model(real_samples)
        fake_pred = self.model(fake_samples)
        return -torch.mean(real_pred) + torch.mean(fake_pred)


class AdversarialLoss(nn.Module):
    """
    Простейший «генераторный» loss для WGAN: 
      −E[D(G(z))]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fake_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
        return -torch.mean(self.model(fake_samples))