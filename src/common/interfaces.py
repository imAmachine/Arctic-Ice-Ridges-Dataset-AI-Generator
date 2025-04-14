from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, Any, List, Type


class IProcessor(ABC):
    """Interface for user defined image processing classes
    
    Attributes:
        NAME: Name identifier for the processor
        PROCESSORS_NEEDED: List of processor classes that should be applied before this processor
    """
    
    def __init__(self, processor_name: str = None):
        self.NAME = processor_name if processor_name else self.__class__.__name__
        self.metadata = {}
        self.VALUE = "False"
        self._result_value = None
    
    @property
    def PROCESSORS_NEEDED(self) -> List[Type['IProcessor']]:
        """Должен быть переопределен в дочерних классах"""
        return []
    
    @abstractmethod
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Метод обработки изображения, реализуемый в дочерних классах
        
        Args:
            image (np.ndarray): Изображение для обработки
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        pass
    
    def get_metadata_value(self) -> str:
        """Возвращает значение для записи в метаданные
        
        По умолчанию возвращает значение _result_value если оно установлено,
        иначе возвращает "True"
        """
        if self._result_value is not None:
            return str(self._result_value)
        return "True"
    
    def check_conditions(self, metadata: Dict[str, str]) -> bool:
        """Проверяет, выполнены ли все необходимые условия для запуска процессора
        
        Args:
            metadata: Текущие метаданные процесса обработки
            
        Returns:
            bool: True, если все необходимые процессоры были выполнены успешно
        """
        for processor_class in self.PROCESSORS_NEEDED:
            processor_name = processor_class.__name__
            proc_val = metadata.get(processor_name)
            if processor_name not in metadata or proc_val in ("False", None):
                print(f'Need {processor_name} before {self.NAME}')
                return False
        return True
    
    def process(self, image: np.ndarray, metadata: Dict[str, str]) -> np.ndarray:
        """Выполняет процесс обработки изображения
        
        Args:
            image: Изображение для обработки
            metadata: Метаданные процесса обработки
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        self.metadata = metadata
        processed_image = image
        
        if self.check_conditions(metadata):
            processed_image = self.process_image(image)
            self.VALUE = self.get_metadata_value()
        else:
            self.VALUE = "False"
            
        metadata[self.NAME] = self.VALUE
        return processed_image


class IModelTrainer(ABC):
    @abstractmethod
    def __init__(self, model, scheduler, optimizer, loss_fn):
        self.model = model
        self.scheduller = scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    @abstractmethod
    def save_model_state_dict(self):
        pass
    
    @abstractmethod
    def step(self):
        pass
