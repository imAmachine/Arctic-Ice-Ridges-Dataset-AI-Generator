from torch import nn, autograd
import torch.nn.functional as F
from torch import Tensor, tensor, rand, mean, ones_like, sigmoid


class GradientPenalty(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fake_samples: Tensor, real_samples: Tensor) -> Tensor:
        real = real_samples.detach()
        fake = fake_samples.detach()
        
        batch = real.size(0)
        alpha = rand(batch, 1, 1, 1, device=real.device)
        interpolates = real + alpha * (fake - real)
        interpolates.requires_grad_(True)

        d_int = self.model(interpolates)
        d_int = d_int.view(batch, -1).mean(dim=1)

        grads = autograd.grad(
            outputs=d_int.sum(),
            inputs=interpolates,
            create_graph=True
        )[0]

        grads = grads.view(batch, -1)
        grad_norm = grads.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp


class WassersteinLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.drift_eps = 1e-3

    def forward(self, fake_samples: Tensor, real_samples: Tensor) -> Tensor:
        real_pred = self.model(real_samples)
        fake_pred = self.model(fake_samples)
        drift = self.drift_eps * (real_pred ** 2).mean()
        
        return fake_pred.mean() - real_pred.mean() + drift


class GeneratorLoss(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fake_samples: Tensor, real_samples: Tensor = None) -> Tensor:
        return -self.model(fake_samples).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.register_buffer('smooth', tensor(smooth))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        prob = sigmoid(pred)
        prob_flat   = prob.view(prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (prob_flat * target_flat).sum(dim=1)
        denom = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"  # "mean" | "sum" | "none"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = sigmoid(pred)
        p_t = target * prob + (1 - target) * (1 - prob)
        
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = alpha_factor * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]])
        self.register_buffer('kernel_x', kx.float())
        self.register_buffer('kernel_y', kx.transpose(2,3).float())

    def forward(self, pred, target):
        gx_gen = F.conv2d(pred, self.kernel_x, padding=1)
        gy_gen = F.conv2d(pred, self.kernel_y, padding=1)
        gx_tgt = F.conv2d(target, self.kernel_x, padding=1)
        gy_tgt = F.conv2d(target, self.kernel_y, padding=1)
        return F.l1_loss(gx_gen, gx_tgt) + F.l1_loss(gy_gen, gy_tgt)
