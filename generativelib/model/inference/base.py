from typing import Dict

import torch
from generativelib.model.train.base import ArchModule


#
#   WIP - доработка идёт
#
class ModuleInference(ArchModule):
    def __init__(self, model_type, module, inference_params: Dict):
        super().__init__(model_type, module)
        self.output_path = inference_params.get("out_path")
        self.weights_path = inference_params.get("weights_path")
        
        if self.weights_path:
            self.module.load_state_dict(self.weights_path)
    
    def generate(self, inp: torch.Tensor) -> None:
        with torch.no_grad():
            self.module(inp)