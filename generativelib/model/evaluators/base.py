from __future__ import annotations
import torch
from torch import nn
from dataclasses import dataclass
from typing import Callable, Dict, List

# enum
from generativelib.model.evaluators.losses import *
from generativelib.model.evaluators.metrics import *
from sklearn.metrics import f1_score, jaccard_score, precision_score

from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.enums import EvaluatorType, LossName, MetricName


LOSSES = {
    LossName.BCE_Logits.value: nn.BCEWithLogitsLoss,
    LossName.BCE.value: nn.BCELoss,
    LossName.L1.value: nn.L1Loss,
    LossName.EDGE.value: EdgeLoss,
    LossName.FOCAL.value: FocalLoss,
    LossName.DICE.value: DiceLoss
}

METRICS = {
    MetricName.PRECISION.value: precision_score,
    MetricName.F1.value: f1_score,
    MetricName.IOU.value: jaccard_score,
    MetricName.FD.value: FractalMetric,
}


@dataclass
class Evaluator:
    """Датакласс для метрики или лосса и нужной логикой """
    callable_fn: Callable
    name: str
    type: str
    weight: float = 1.0
    exec_phase: str = ExecPhase.ANY.value
    
    def __call__(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> 'torch.Tensor':
        return self.callable_fn(generated_sample, real_sample) * self.weight


class EvaluateProcessor:
    """Нужен для подсчёта метрик и лоссов"""
    def __init__(self, device: torch.device, evaluators: List[Evaluator]):
        self.device = device
        self.evaluators = evaluators
        self.evaluators_history: Dict[ExecPhase, List] = None
        self._init_history_dict()

    def _init_history_dict(self):
        self.evaluators_history = {
            ExecPhase.TRAIN.value: [], 
            ExecPhase.VALID.value: []
        }
    
    def _update_history(self, eval_type: str, phase: str, name: str, value: 'torch.Tensor') -> None:
        current_history = self.evaluators_history[phase][-1]
        current_history[eval_type][name] = value

    def process(self, generated_sample, real_sample, exec_phase: str) -> None:
        evaluators_types = [
            member.value
            for member in EvaluatorType
        ]

        self.evaluators_history[exec_phase].append({
            e_type: {} 
            for e_type in evaluators_types
        })
        
        for item in self.evaluators:
            
            if item.exec_phase == ExecPhase.ANY.value or item.exec_phase == exec_phase:
                weighted_tensor = item(generated_sample, real_sample)
                
                self._update_history(
                    eval_type=item.type, 
                    phase=exec_phase, 
                    name=item.name, 
                    value=weighted_tensor
                )
    
    def compute_epoch_summary(self) -> Dict[ExecPhase, Dict[str, Dict[str, float]]]:
        full_result = {}

        for phase, phase_history in self.evaluators_history.items():
            if not phase_history:
                full_result[phase] = {}
                continue

            summary = {}

            for step_result in phase_history:
                for eval_type, evaluators in step_result.items():
                    for name, val in evaluators.items():
                        summary.setdefault(eval_type, {}) \
                        .setdefault(name, []) \
                        .append(val.clone().detach())

            averaged = {
                eval_type: {
                    name: torch.stack(values).mean().item()
                    for name, values in metrics.items()
                }
                for eval_type, metrics in summary.items()
            }

            full_result[phase] = averaged

        return full_result
