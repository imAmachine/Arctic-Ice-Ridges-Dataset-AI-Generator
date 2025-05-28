from abc import ABC, abstractmethod
from typing import Dict, Iterator, List
import torch


class BaseMaskProcessor(ABC):    
    @abstractmethod
    def _realization(self, cloned_mask: torch.Tensor) -> None:
        pass
    
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        new_mask = mask.clone()
        self._realization(new_mask)
        return new_mask


class MaskProcessorsFabric:
    def __init__(self):
        self.processors: List[BaseMaskProcessor] = []
    
    def create_processors(self, processors_dict: Dict):
        from generativelib.dataset.mask_processors import MASK_PROCESSORS
        self.processors.clear()
        
        for name, values in processors_dict.items():
            if values["enabled"]:
                cls = MASK_PROCESSORS.get(name)
                self.processors.append(cls(**values["params"]))

        if not self.processors:
            raise RuntimeError("[masking] Ни одного валидного процессора не создано")
    
    def __iter__(self) -> Iterator[BaseMaskProcessor]:
        """Возвращаем итератор по внутреннему списку процессоров."""
        return iter(self.processors)
    
    