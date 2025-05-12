import torch

from src.common.analyze_tools import FractalAnalyzerGPU
from src.common.structs import TrainPhases as phases


# WIP ====== заготовка
class Loss:
    def __init__(self, loss_fn, loss_name, loss_weight, phase: phases, loss_args: tuple):
        self.criterion = loss_fn
        self.name = loss_name
        self.loss_weight = loss_weight
        self.phase = phase
        self.args = loss_args


def gradient_penalty(model, device, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates.requires_grad_(True)

        d_interpolates = model(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

def wasserstein_loss(model, real_sample, fake_sample):
    real_pred, fake_pred = model(real_sample), model(fake_sample)
    return -real_pred.mean() + fake_pred.mean()

def adversarial_loss(model, input_data):
    return -model(input_data).mean()

def calc_fractal_loss(generated: torch.Tensor, target: torch.Tensor) -> float:
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