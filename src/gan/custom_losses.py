from torch.autograd import grad
import torch.nn as nn
import torch


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