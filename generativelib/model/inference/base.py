from typing import Dict, Self

import torch
from generativelib.model.common.checkpoint import CheckpointManager
from generativelib.model.train.base import ArchModule


#
#   WIP - доработка идёт
#
class ModuleInference(ArchModule):
    def __init__(self, model_type, module):
        super().__init__(model_type, module)
        self.model_type = model_type

    def load_weights(self, weights_path: str):
        CheckpointManager.load_state(self, weights_path)

    def from_state_dict(self, state_dict: Dict) -> Self:
        module_weights = state_dict[self.model_type.name.lower()]["module_state"]["module"]
        self.module.load_state_dict(module_weights)
        return self
    
    def generate(self, *args, **kwargs) -> None:
        with torch.no_grad():
            return self.module(*args, **kwargs)
