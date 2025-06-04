from __future__ import annotations
from typing import Dict
import torch

# Enums
from generativelib.model.arch.enums import GenerativeModules


class ArchModule(torch.nn.Module):
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
    def from_dict(cls, module_name: str, arch_params: Dict):
        module_cls = GenerativeModules[module_name.upper()].value
        arch_module = module_cls(**arch_params)
        
        return cls(GenerativeModules[module_name.upper()], arch_module)
    
    def forward(self, x):
        return self.module(x)