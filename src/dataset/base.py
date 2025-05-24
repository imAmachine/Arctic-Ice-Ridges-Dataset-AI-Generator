from abc import ABC, abstractmethod
import torch


class BaseProcessStrategy(ABC):    
    @abstractmethod
    def _realization(self, mask: torch.Tensor):
        pass
    
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:        
        return self._realization(mask)