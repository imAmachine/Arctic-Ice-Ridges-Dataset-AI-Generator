from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class MaskRegion:
    mask: torch.Tensor
    top: int
    left: int
    bh: int
    bw: int


@dataclass
class BaseProcessStrategy(ABC):
    def __init__(self, strategy_name: str):
        self.name = strategy_name
    
    @abstractmethod
    def apply(self, mask_region: MaskRegion, mask_params: Dict):
        pass


class ProcessingStrategies(List[BaseProcessStrategy]):
    def apply_all(self, mask_region: MaskRegion, params_dict: Dict):
        for strategy in self:
            params = params_dict.get(strategy.name)
            if params is not None:
                strategy.apply(mask_region, params)