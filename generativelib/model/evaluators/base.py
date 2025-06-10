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
    callable_fn: torch.nn.Module
    name: str
    type: EvaluatorType
    weight: float = 1.0
    exec_phase: ExecPhase = ExecPhase.ANY
    
    @classmethod
    def from_dict(cls, eval_name: str, eval_info: Dict):
        from generativelib.config_tools.default_values import EXEC_PHASE_KEY, EXECUTION_KEY, INIT_KEY
        exec_params = eval_info[EXECUTION_KEY]
        init_params = eval_info[INIT_KEY]
        
        eval_type = EvaluatorType.LOSS
        if eval_name in MetricName:
            eval_type = EvaluatorType.METRIC
        
        if exec_params["weight"] > 0.0:
            cls = LOSSES[eval_name]
            
            eval_func = cls(**init_params)
            exec_phase = ExecPhase[exec_params[EXEC_PHASE_KEY]]
            eval_weight = exec_params["weight"]
            
            return cls(
                callable_fn=eval_func,
                name=eval_name,
                type=eval_type,
                exec_phase=exec_phase,
                weight=eval_weight
            )
    
    def __call__(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> 'torch.Tensor':
        return self.callable_fn(generated_sample, real_sample) * self.weight

    def to(self, device: torch.device):
        self.callable_fn.to(device)


class EvalsCollector:
    # [CLASS AI GENERATED]
    def __init__(self):
        self.reset_history()

    def reset_history(self):
        self.epoch_history = {
            ExecPhase.TRAIN: {},
            ExecPhase.VALID: {},
        }

    def collect(self, values: List[tuple], exec_phase: ExecPhase) -> None:
        if exec_phase not in self.epoch_history:
            self.epoch_history[exec_phase] = {}
        
        phase_hist = self.epoch_history[exec_phase]
        
        for typ, name, val in values:
            key = (typ, name)
            phase_hist.setdefault(key, []).append(val)

    def compute_epoch_summary(self) -> Dict:
        summary = {}
        for phase, phase_hist in self.epoch_history.items():
            for key, history in phase_hist.items():
                count = len(history)
                if count == 0:
                    summary[(phase, *key)] = None
                else:
                    summary[(phase, *key)] = sum(history) / count
        return summary
