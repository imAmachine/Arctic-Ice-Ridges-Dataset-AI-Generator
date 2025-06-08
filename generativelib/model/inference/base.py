from typing import Dict

import torch
from generativelib.model.train.base import ArchModule


#
#   WIP - доработка идёт
#
class ModuleInference(ArchModule):
    def __init__(self, model_type, module):
        super().__init__(model_type, module)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_weights(self, weights_path: str):
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
            model_key = self.model_type.name.lower()
            state = state[model_key]["module_state"]
            self.load_state_dict(state)
    
    def generate(self, *args, **kwargs) -> None:
        with torch.no_grad():
            return self.module(*args, **kwargs)