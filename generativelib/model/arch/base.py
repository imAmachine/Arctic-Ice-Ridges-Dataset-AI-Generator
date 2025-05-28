from dataclasses import dataclass
from typing import List

import torch

from generativelib.model.arch.enums import ModelTypes
from generativelib.model.evaluators.base import Evaluator


@dataclass
class Architecture:
    model_type: ModelTypes
    arch: 'torch.nn.Module'
    optimizer: 'torch.optim.Optimizer'
    scheduler: 'torch.optim.lr_scheduler.LRScheduler'
    evaluators: List[Evaluator]
    
    def __call__(self, inp):
        return self.arch(inp)