from abc import ABC, abstractmethod
from typing import Dict, Self


class ITorchState(ABC):
    @abstractmethod
    def to_state_dict(self) -> Dict:
        pass
    
    @abstractmethod
    def from_state_dict(self, state_dict: Dict) -> Self:
        pass
