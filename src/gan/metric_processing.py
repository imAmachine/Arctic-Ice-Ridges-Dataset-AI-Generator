import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List
from collections import defaultdict
from src.common.structs import ExecPhases as phases, MetricsNames as metrics

@dataclass
class Metric:
    metric_fn: Callable
    name: str
    weight: float
    only_on: phases = phases.ANY

    def __call__(self, generated_sample, real_sample) -> torch.Tensor:    
        if self.weight == 1.0:
            return self.metric_fn(generated_sample, real_sample)

class MetricProcessor:
    def __init__(self, device):
        self.device = device
        self.metrics_list: List['Metric'] = []
        
        self.metric_history = {
            phases.TRAIN: defaultdict(list),
            phases.VALID: defaultdict(list)
        }

    def _prepare_inputs(self, generated, real, mask):
        gen_np = (generated > 0.5).byte().cpu().numpy().flatten()
        real_np = real.byte().cpu().numpy().flatten()
        mask_np = mask.bool().cpu().numpy().flatten()
        return gen_np[mask_np], real_np[mask_np]
    
    
    def calc_metrics(self, generated, real, mask, phase: phases):
        results = {}
        gen_np, real_np = self._prepare_inputs(generated, real, mask)
        for metric in self.metrics_list:
            if metric.only_on == phases.ANY or metric.only_on == phase:
                metric_value = metric(gen_np, real_np)
                results[metric.name] = metric_value
                self.metric_history[phase][metric.name].append(metric_value)
        return results
    
    def new_metric(self, metric: Metric):
        self.metrics_list.append(metric)

    def new_metrics(self, metrics: List[Metric]):
        self.metrics_list.extend(metrics)

    def reset_metrics(self):
        self.loss_history = {phases.TRAIN: [], phases.VALID: []}

    def epoch_avg_metrics(self, phase: phases) -> Dict[str, float]:
        return {
            name: float(np.mean(values)) 
            for name, values in self.metric_history[phase].items()
        }

    def metrics_stringify(self, metrics_dict: Dict[str, float]) -> str:
        if not metrics_dict:
            return "No metrics data available."

        name_w = max(len(name) for name in metrics_dict.keys())
        val_w = max(len(f"{val:.4f}") for val in metrics_dict.values())

        lines = [
            '\nAverage Metrics:',
            f"+-{'-'*name_w}-+-{'-'*val_w}-+",
            f"| {'Metric'.ljust(name_w)} | {'Value'.rjust(val_w)} |",
            f"+-{'-'*name_w}-+-{'-'*val_w}-+"
        ]
        
        for name, val in metrics_dict.items():
            lines.append(f"| {name.ljust(name_w)} | {val:>{val_w}.4f} |")
        
        lines.append(f"+-{'-'*name_w}-+-{'-'*val_w}-+")
        return '\n'.join(lines)

    def epoch_metrics_str(self, phase: phases) -> str:
        return self.metrics_stringify(self.epoch_avg_metrics(phase))