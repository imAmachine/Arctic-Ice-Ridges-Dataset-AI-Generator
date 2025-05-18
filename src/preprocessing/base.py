from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type


class Processor(ABC):
    """Интерфейс процессора изображения."""

    dependencies: List[Type["Processor"]] = []

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(self, image: Any, metadata: Dict[str, Any]) -> Any:
        pass

    def get_metadata_value(self) -> str:
        return "True"
