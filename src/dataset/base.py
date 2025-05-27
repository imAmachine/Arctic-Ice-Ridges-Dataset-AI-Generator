from abc import ABC, abstractmethod
import torch


class BaseProcessStrategy(ABC):    
    @abstractmethod
    def _realization(self, cloned_mask: torch.Tensor) -> None:
        pass
    
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        new_mask = mask.clone()
        self._realization(new_mask)
        return new_mask