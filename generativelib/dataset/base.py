from abc import ABC, abstractmethod
from typing import Dict, Type
import torch




class BaseMaskProcessor(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass
    
    @abstractmethod
    def _realization(self, cloned_mask: torch.Tensor) -> None:
        pass
    
    @classmethod
    def from_dict(cls, name: str, values: Dict):
        from generativelib.config_tools.default_values import PARAMS_KEY, ENABLED_KEY
        from generativelib.dataset.mask_processors import MASK_PROCESSORS
        
        if not values.get(ENABLED_KEY, False):
            return None

        if name not in MASK_PROCESSORS:
            raise KeyError(f"Неизвестный mask-processor с именем '{name}'")

        proc_cls: Type[BaseMaskProcessor] = MASK_PROCESSORS[name]
        params = values.get(PARAMS_KEY, {})

        return proc_cls(**params)
    
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        new_mask = mask.clone()
        self._realization(new_mask)
        return new_mask
    