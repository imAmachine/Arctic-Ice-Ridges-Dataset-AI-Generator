from __future__ import annotations
from typing import Dict, Self
import torch

# Enums
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.common.interfaces import ITorchState


class ArchModule(torch.nn.Module, ITorchState):
    """Обёртка над torch.nn.Module, позволяет легко инициализировать и идентифицировать нужный модуль по model_type"""
    def __init__(
        self,
        model_type: GenerativeModules,
        module: torch.nn.Module
    ):
        super().__init__()
        self.model_type = model_type
        self.module = module

    @classmethod
    def cls_from_dict(cls, module_name: str, arch_params: Dict):
        module_cls = GenerativeModules[module_name.upper()].value
        arch_module = module_cls(**arch_params)
        
        return cls(GenerativeModules[module_name.upper()], arch_module)
    
    def to_state_dict(self) -> Dict:
        return {
            "model_type": self.model_type.name,
            "module": self.module.state_dict()
        }
    
    def from_state_dict(self, state_dict: Dict) -> Self:
        self.model_type = GenerativeModules[state_dict["model_type"]]
        self.module.load_state_dict(state_dict["module"])
        return self
    
    def forward(self, x):
        return self.module(x)