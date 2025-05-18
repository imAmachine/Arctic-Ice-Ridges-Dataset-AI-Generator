from dataclasses import dataclass, field

from typing import Callable, Dict, List
import torch

from src.common.enums import ExecPhase, EvaluatorType

@dataclass
class Evaluator:
    """Датакласс для метрики или лосса и нужной логикой """
    callable_fn: Callable
    name: str
    type: str
    weight: float = field(default=1.0)
    exec_phase: str = ExecPhase.ANY.value
    
    def __call__(self, generated_sample, real_sample) -> torch.Tensor:   
        if self.weight <= 0.0:
            return torch.tensor(0.0, device=generated_sample.device)
        return self.callable_fn(generated_sample, real_sample) * self.weight


class EvalProcessor:
    """Нужен для подсчёта метрик и лоссов внутри ModuleTrainer"""
    def __init__(self, device: torch.device, evaluators: List[Evaluator] = []):
        self.device = device
        self.evaluators = evaluators
        self.evaluators_history: Dict[ExecPhase, List] = {
            ExecPhase.TRAIN: [], ExecPhase.VALID: []
        }

    def _update_history(self, eval_type: str, phase: ExecPhase, name: str, value: 'torch.Tensor') -> None:
        current_history = self.evaluators_history[phase][-1]
        current_history[eval_type].update({name: value})

    def process(self, generated_sample, real_sample, exec_phase: str) -> None:
        evaluators_types = [member.value for member in EvaluatorType]
        self.evaluators_history[exec_phase].append({e_type: {} for e_type in evaluators_types})
        
        for item in self.evaluators:
            phase_value = exec_phase.value
            if item.exec_phase == ExecPhase.ANY.value or item.exec_phase == phase_value:
                weighted_tensor = item(generated_sample, real_sample)
                self._update_history(item.type, exec_phase, item.name, weighted_tensor)

    def reset_history(self):
        self.evaluators_history = {ExecPhase.TRAIN: [], ExecPhase.VALID: []}
    
    def compute_epoch_summary(self) -> Dict[ExecPhase, Dict[str, Dict[str, float]]]:
        """
        Вычисляет средние значения метрик за эпоху для всех фаз (TRAIN, VALID и т.д.).
        
        Returns:
            Словарь:
            {
                ExecPhase.TRAIN: {
                    "eval_type": { "metric_name": mean_value, ... },
                    ...
                },
                ExecPhase.VALID: { ... }
            }
        """
        full_result = {}

        for phase, phase_history in self.evaluators_history.items():
            if not phase_history:
                full_result[phase] = {}
                continue

            summary = {}

            for step_result in phase_history:
                for eval_type, metrics in step_result.items():
                    for name, val in metrics.items():
                        summary.setdefault(eval_type, {}).setdefault(name, []).append(val.clone().detach())

            averaged = {
                eval_type: {
                    name: torch.stack(values).mean().item()
                    for name, values in metrics.items()
                }
                for eval_type, metrics in summary.items()
            }

            full_result[phase] = averaged

        return full_result

