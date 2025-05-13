from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np
import torch

from src.common.analyze_tools import FractalAnalyzerGPU
from src.common.structs import ExecPhases as phases, LossNames as losses


@dataclass
class Loss:
    loss_fn: Callable
    name: str
    weight: float
    only_on: phases = phases.ANY

    def __call__(self, generated_sample, real_sample) -> torch.Tensor:      
        if self.weight <= 0.0:
            return torch.tensor(0.0, device=generated_sample.device)
        return self.loss_fn(generated_sample, real_sample) * self.weight


class LossProcessor:
    def __init__(self, device):
        self.device = device
        self.losses_list: List['Loss'] = []
        
        self.loss_history = {phases.TRAIN: [], phases.VALID: []}
    
    def _update_loss_history(self, phase: phases, loss_n: str, loss_v: 'torch.Tensor') -> None:
        current_history = self.loss_history[phase][-1]
        current_history.update({loss_n: loss_v})
        current_history[losses.TOTAL.value] = current_history.get(losses.TOTAL.value, 0.0) + loss_v
    
    def calc_losses(self, real_sample, generated_sample, phase: phases) -> 'torch.Tensor':
        self.loss_history[phase].append({})
        total_train_loss = torch.tensor(0.0, device=self.device)
        
        # расчёт лоссов
        for loss in self.losses_list:
            if loss.only_on == phases.ANY or loss.only_on == phase:
                loss_tensor = loss(generated_sample, real_sample)
                
                if phase == phases.TRAIN:
                    total_train_loss = total_train_loss + loss_tensor
                
                self._update_loss_history(phase, loss.name, loss_tensor.item())
        
        return total_train_loss

    def new_loss(self, loss: Loss):
        self.losses_list.append(loss)
    
    def new_losses(self, losses: List[Loss]):
        self.losses_list.extend(losses)

    def reset_losses(self):
        self.loss_history = {phases.TRAIN: [], phases.VALID: []}
        
    def batch_avg_losses(self, aggregated: Dict[str, list]) -> Dict[str, float]:
        return {name: float(np.mean(vals)) for name, vals in aggregated.items()}

    def epoch_avg_losses(self, phase: phases, batch_size: int) -> Dict[str, float]:
        history = self.loss_history.get(phase, [])
        if not history:
            return {}
        recent = history[-batch_size:]
        aggregated: Dict[str, list] = defaultdict(list)
        for batch_losses in recent:
            for loss_name, loss_val in batch_losses.items():
                aggregated[loss_name].append(loss_val)
        return self.batch_avg_losses(aggregated)

    def losses_stringify(self, losses_dict: Dict[str, float]) -> str:
        if not losses_dict:
            return "No loss data available."

        total_val = losses_dict.get(losses.TOTAL.value, None)
        rows = [(k, v) for k, v in losses_dict.items() if k != losses.TOTAL.value]
        
        if total_val is not None:
            rows.append((losses.TOTAL.value, total_val))

        name_w = max(len(name) for name, _ in rows + [("Loss", 0)])
        val_w  = max(len(f"{val:.4f}") for _, val in rows + [("", 0.0)])

        lines = []
        lines.append('\n')
        lines.append(f"Average losses for {self.__class__.__name__}:")
        lines.append(f"+-{'-'*name_w}-+-{'-'*val_w}-+")
        lines.append(f"| {'Loss'.ljust(name_w)} | {'Value'.rjust(val_w)} |")
        lines.append(f"+-{'-'*name_w}-+-{'-'*val_w}-+")

        for name, val in rows:
            lines.append(f"| {name.ljust(name_w)} | {val:>{val_w}.4f} |")

        lines.append(f"+-{'-'*name_w}-+-{'-'*val_w}-+")
        return "\n".join(lines)

    def epoch_avg_losses_str(self, phase: phases, batch_size: int) -> str:
        avg_losses = self.epoch_avg_losses(phase, batch_size)
        return self.losses_stringify(avg_losses)

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
    