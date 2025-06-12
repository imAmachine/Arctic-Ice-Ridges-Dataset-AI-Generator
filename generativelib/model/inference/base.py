from typing import Dict

import torch
from generativelib.model.common.checkpoint import CheckpointManager
from generativelib.model.train.base import ArchModule


#
#   WIP - доработка идёт
#
class ModuleInference(ArchModule):
    def __init__(self, model_type, module):
        super().__init__(model_type, module)

    def load_weights(self, weights_path: str):
        CheckpointManager.load_state(self, weights_path)
    
    def generate(self, *args, **kwargs) -> None:
        with torch.no_grad():
            return self.module(*args, **kwargs)
