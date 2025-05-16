from dataclasses import dataclass, field

from typing import Callable, Dict, List
import torch
from src.common.structs import ExecPhase as phases, LossName as losses, EvaluatorType as eval_type

@dataclass
class Evaluator:
    callable_fn: Callable
    name: str
    type: str
    weight: float = field(default=1.0)
    exec_phase: str = phases.ANY.value
    
    def __call__(self, generated_sample, real_sample) -> torch.Tensor:   
        if self.weight <= 0.0:
            return torch.tensor(0.0, device=generated_sample.device)
        return self.callable_fn(generated_sample, real_sample) * self.weight


@dataclass
class EvalProcessor:
    device: torch.device
    evaluators: List[Evaluator] = field(default_factory=list)
    evaluators_history: Dict[phases, List] = field(default_factory=lambda: {
        phases.TRAIN: [], phases.VALID: []
    })

    def _update_history(self, eval_type: str, phase: phases, name: str, value: 'torch.Tensor') -> None:
        current_history = self.evaluators_history[phase][-1]
        current_history[eval_type].update({name: value})

    def process(self, generated_sample, real_sample, exec_phase: str) -> None:
        evaluators_types = [member.value for member in eval_type]
        self.evaluators_history[exec_phase].append({e_type: {} for e_type in evaluators_types})
        
        for item in self.evaluators:
            phase_value = exec_phase.value
            if item.exec_phase == phases.ANY.value or item.exec_phase == phase_value:
                weighted_tensor = item(generated_sample, real_sample)
                self._update_history(item.type, exec_phase, item.name, weighted_tensor)

    def reset_history(self):
        self.evaluators_history = {phases.TRAIN: [], phases.VALID: []}
    
    def compute_epoch_summary(self, phase: phases) -> Dict[str, Dict[str, float]]:
        history = self.evaluators_history[phase]
        if not history:
            return {}

        summary = {}
        
        for step_result in history:
            for eval_type_key, values in step_result.items():
                if eval_type_key not in summary:
                    summary[eval_type_key] = {}
                
                for name, val in values.items():
                    if name not in summary[eval_type_key]:
                        summary[eval_type_key][name] = []
                    summary[eval_type_key][name].append(val.clone().detach())

        # Среднее значение
        result = {}
        for eval_type_key, metrics in summary.items():
            result[eval_type_key] = {name: torch.stack(vals).mean().item() for name, vals in metrics.items()}

        return result
    
    def print_eval_summary(self, name: str, phase: phases):
        summary = self.compute_epoch_summary(phase=phase)

        for group, metrics in summary.items():
            print('\n')
            if len(metrics.items()) > 0:
                print(f"[{name}] {group.upper()}:")
                for k, v in metrics.items():
                    print(f"\t{k}: {v:.4f}")

