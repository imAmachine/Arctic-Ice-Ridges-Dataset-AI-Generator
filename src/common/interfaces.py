from abc import ABC, abstractmethod
import os
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List, Type
from tabulate import tabulate
import pandas as pd
import torch
from src.common.structs import ExecPhase, ModelType

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
                raise Exception(f'Need {processor_name} before {self.NAME}')
    
    def process(self, image: np.ndarray, metadata: Dict[str, str]) -> np.ndarray:
        """Выполняет процесс обработки изображения
        
        Args:
            image: Изображение для обработки
            metadata: Метаданные процесса обработки
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        self.metadata = metadata
        
        self.check_conditions(metadata)
        processed_image = self.process_image(image)
        self.VALUE = self.get_metadata_value()
    
        metadata[self.NAME] = self.VALUE        
        return processed_image


class IGenerativeModel(ABC):
    def __init__(self, 
                 device: 'torch.device', 
                 modules: List,
                 output_path: str):
        from src.gan.model import ModuleTrainManager
        self.modules: List = [module.arch.to(device) for module in modules]
        self.t_managers: 'Dict[ModelType, ModuleTrainManager]' = {m.model_type: ModuleTrainManager(device, m) for m in modules}
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
    
    @abstractmethod
    def _training_pipeline(self, input_data, target_data):
        pass
    
    @abstractmethod
    def _validation_pipeline(self, input_data, target_data):
        pass
    
    @abstractmethod
    def _evaluation_pipeline(self, input_data, target_data):
        pass
    
    def model_step(self, input_data, target_data, phase: ExecPhase) -> None:
        self.set_phase(phase)
        
        if phase == ExecPhase.TRAIN:
            self._training_pipeline(input_data, target_data)
            
        if phase == ExecPhase.VALID or phase == ExecPhase.EVAL:
            with torch.no_grad():
                if phase == ExecPhase.VALID:
                    self._validation_pipeline(input_data, target_data)
                if phase == ExecPhase.EVAL:
                    self._evaluation_pipeline(input_data, target_data)
    
    def __merge_epoch_summaries(
        self,
        summaries: list[dict[str, dict[str, float]]],
        names: list[str]
    ) -> pd.DataFrame:
        all_eval_types = set()
        for s in summaries:
            all_eval_types |= s.keys()

        rows = []
        for et in sorted(all_eval_types):
            all_metrics = set()
            for s in summaries:
                all_metrics |= s.get(et, {}).keys()

            for m in sorted(all_metrics):
                row = {"Eval Type": et, "Metric": m}
                
                for name, s in zip(names, summaries):
                    row[name] = s.get(et, {}).get(m, float("nan"))
                rows.append(row)

        df = pd.DataFrame(rows).set_index(["Eval Type", "Metric"])
        return df
    
    def print_eval_summary(self, phase: ExecPhase):
        computed = {}
        for m_type, manager in self.t_managers.items():
            computed[m_type.value] = manager.evaluate_processor.compute_epoch_summary(phase)
        
        names = list(computed.keys())
        summaries = list(computed.values())
        df = self.__merge_epoch_summaries(summaries, names)
        df_reset = df.reset_index()
        print(tabulate(df_reset, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
    
    def clear_eval_history(self):
        for _, manager in self.t_managers.items():
            manager.evaluate_processor.reset_history()
    
    def set_phase(self, phase: ExecPhase=ExecPhase.TRAIN) -> None:
        for _, manager in self.t_managers.items():
            if phase == ExecPhase.TRAIN:
                manager.module.arch.train()
            
            if phase == ExecPhase.VALID or phase == ExecPhase.EVAL:
                manager.module.arch.eval()
    
    def save_batch_plot(self, input_data, target_data, gen_data, samples_count: int = 1, phase: ExecPhase = ExecPhase.TRAIN):
        n_cols = min(5, samples_count)

        plt.figure(figsize=(12, 12), dpi=300)

        batches = [input_data, gen_data, target_data]
        row_titles = ['Input', 'Generated', 'Target']

        for i, batch in enumerate(batches):
            for j in range(n_cols):
                img = batch[j]
                ax_index = i * n_cols + j + 1
                plt.subplot(3, n_cols, ax_index)
                plt.imshow(img.detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
                plt.title(f"{row_titles[i]} {j+1}", fontsize=12, pad=10)
                plt.axis('off')

        plt.suptitle(f'Phase: {phase}', fontsize=14, y=1.02)
        plt.tight_layout(pad=3.0)

        output_file = f"{self.output_path}/{phase.value}.png"
        plt.savefig(output_file)
        plt.close()
