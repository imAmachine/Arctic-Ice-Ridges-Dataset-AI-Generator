from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from typing import Dict, Any, List, Literal, Type


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


class IGenerativeModel:
    def __init__(self, target_image_size, device, optimization_params: Dict):
        self.target_image_size = target_image_size
        self.device = device
        self.optimization_params = optimization_params
    
    @abstractmethod
    def switch_mode(self, mode: Literal['train', 'valid'] = 'train') -> None:
        pass
    
    @abstractmethod
    def save_checkpoint(self, output_path):
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        pass
    
    @abstractmethod
    def train_step(self, **args):
        pass
    
    @abstractmethod
    def eval_step(self, **args):
        pass
    
    @abstractmethod
    def step_schedulers(self, metric: str):
        pass

class IModelTrainer(ABC):
    def __init__(self, model, losses_weights: Dict):
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.criterion = self._calc_losses
        self.losses_history = {'train': [], 'valid': []}
        self.losses_weights = losses_weights
    
    @abstractmethod
    def _calc_losses(self):
        pass
    
    @abstractmethod
    def train_step(self):
        pass
    
    @abstractmethod
    def eval_step(self):
        pass
    
    def step_scheduler(self, metric):
        self.scheduler.step(metric)
    
    def reset_losses(self):
        self.losses_history = {'train': [], 'valid': []}
    
    def epoch_avg_losses_str(self, phase: Literal['train', 'valid'], batch_size: int) -> str:
        losses_history = self.losses_history[phase]

        if losses_history:
            epoch_losses = losses_history[-batch_size:]

            # сбор значений каждой метрики по каждому батчу
            aggregated_losses = defaultdict(list)
            for batch_losses in epoch_losses:
                for loss_name, loss_value in batch_losses.items():
                    aggregated_losses[loss_name].append(loss_value)

            # расчёт среднего значения за эпоху
            avg_trainer_losses = {loss_n: np.mean(loss_v) for loss_n, loss_v in aggregated_losses.items()}

            losses_str = f'Average losses for {self.__class__.__name__}:\n'
            for loss_n, loss_v in avg_trainer_losses.items():
                losses_str += f'\t{loss_n}: {loss_v:.4f}\n'

            return losses_str

        return "No loss data available."