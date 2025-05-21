from abc import ABC, abstractmethod
import torch


class BaseProcessStrategy(ABC):
    def __init__(self, strategy_name: str):
        self.name = strategy_name
    
    @abstractmethod
    def _realization(self, mask: torch.Tensor):
        pass
    
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:        
        return self._realization(mask.clone())