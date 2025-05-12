from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from typing import Dict, List, Literal, Type
import torch

from src.common.structs import TrainPhases as phases


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
    def valid_step(self, **args):
        pass
    
    @abstractmethod
    def step_schedulers(self, metric: str):
        pass

class IModelTrainer(ABC):
    def __init__(self, model, device, losses_weights: Dict):
        self.model = model
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
        self.criterion = self._losses
        self.loss_history = {phases.TRAIN: [], phases.VALID: []}
        self.loss_weights = losses_weights
        
        self.total_train_loss = torch.tensor(0.0, device=self.device)
    
    @abstractmethod
    def _losses(self):
        pass
    
    @abstractmethod
    def train_step(self):
        pass
    
    @abstractmethod
    def eval_step(self):
        pass
    
    def train_step(self, samples: tuple) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        
        self.total_train_loss = torch.tensor(0.0, device=self.device)
        self.loss_history[phases.TRAIN].append({})
        
        self.criterion(*samples, phase=phases.TRAIN)
        self.total_train_loss.backward()
        
        self.optimizer.step()
    
    def eval_step(self, samples: tuple) -> None:
        self.model.eval()
        self.loss_history[phases.VALID].append({})
        
        self.criterion(*samples, phase=phases.VALID)
    
    def _update_loss_history(self, phase: phases, loss_n: str, loss_v: 'torch.tensor'):
        current_history = self.loss_history[phase][-1]
        current_history.update({loss_n: loss_v})
        current_history['total'] = current_history.get('total', 0.0) + loss_v
    
    def calc_loss(self, loss_fn, loss_name, phase: phases, args: tuple) -> Dict[str, float]:
        weight = self.loss_weights.get(loss_name, 1.0)
        loss_tensor = loss_fn(*args) * weight
        
        if phase == phases.TRAIN:
            self.total_train_loss = self.total_train_loss + loss_tensor
        
        self._update_loss_history(phase, loss_name, loss_tensor.item())
    
    def step_scheduler(self, metric):
        self.scheduler.step(metric)
    
    def reset_losses(self):
        self.loss_history = {phases.TRAIN: [], phases.VALID: []}
    
    def batch_avg_losses(self, aggregated: Dict[str, list]) -> Dict[str, float]:
        return {name: float(np.mean(vals)) for name, vals in aggregated.items()}

    def epoch_avg_losses(self, phase: phases, batch_size: int) -> Dict[str, float]:
        history = self.loss_history.get(phase, [])
        if not history:
            return {}
        recent = history[-batch_size:]
        aggregated: Dict[str, list] = defaultdict(list)
        for batch_losses in recent:
            for loss_name, loss_val in batch_losses.items():
                aggregated[loss_name].append(loss_val)
        return self.batch_avg_losses(aggregated)

    def losses_stringify(self, losses: Dict[str, float]) -> str:
        if not losses:
            return "No loss data available."
        s = f'Average losses for {self.__class__.__name__}:\n'
        for loss_name, loss_val in losses.items():
            s += f'\t{loss_name}: {loss_val:.4f}\n'
        return s

    def epoch_avg_losses_str(self, phase: phases, batch_size: int) -> str:
        avg_losses = self.epoch_avg_losses(phase, batch_size)
        return self.losses_stringify(avg_losses)