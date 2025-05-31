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
    LossName.BCE_Logits.name: nn.BCEWithLogitsLoss,
    LossName.BCE.name: nn.BCELoss,
    LossName.L1.name: nn.L1Loss,
    LossName.EDGE.name: EdgeLoss,
    LossName.FOCAL.name: FocalLoss,
    LossName.DICE.name: DiceLoss
}

METRICS = {
    MetricName.PRECISION.name: precision_score,
    MetricName.F1.name: f1_score,
    MetricName.IOU.name: jaccard_score,
    MetricName.FRACTAL_DIMENSION.name: FractalMetric,
}


@dataclass
class EvalItem:
    """Датакласс для метрики или лосса и нужной логикой """
    callable_fn: Callable
    name: str
    type: EvaluatorType
    weight: float = 1.0
    exec_phase: str = ExecPhase.ANY
    
    def __call__(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> 'torch.Tensor':
        return self.callable_fn(generated_sample, real_sample) * self.weight


class EvaluateProcessor:
    """Нужен для подсчёта метрик и лоссов"""
    def __init__(self, evals: List[EvalItem]):
        self.evals_history: Dict[ExecPhase, List] = None
        self.evals = evals
        self._init_history_dict()

    def _init_history_dict(self):
        self.evals_history = {
            ExecPhase.TRAIN: [], 
            ExecPhase.VALID: []
        }
    
    def _update_history(self, eval_type: EvaluatorType, phase: ExecPhase, name: str, value: 'torch.Tensor') -> None:
        current_history = self.evals_history[phase][-1]
        current_history[eval_type][name] = value

    def process(self, generated_sample, real_sample, exec_phase: ExecPhase) -> None:
        evals_types = [member for member in EvaluatorType]

        self.evals_history[exec_phase].append({
            e_type: {} 
            for e_type in evals_types
        })
        
        for item in self.evals:
            if item.exec_phase == ExecPhase.ANY or item.exec_phase == exec_phase:
                weighted_tensor = item(generated_sample, real_sample)

                self._update_history(
                    eval_type=item.type, 
                    phase=exec_phase, 
                    name=item.name, 
                    value=weighted_tensor
                )
    
    def compute_epoch_summary(self) -> Dict[ExecPhase, Dict[str, Dict[str, float]]]:
        full_result = {}

        for phase, phase_history in self.evals_history.items():
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
