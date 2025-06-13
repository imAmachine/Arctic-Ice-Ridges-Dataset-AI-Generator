from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, Type, cast
import torch
import torchvision.transforms.v2
from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate
from src.config_deserializer import TrainConfigDeserializer
from generativelib.dataset.loader import DatasetCreator
from generativelib.model.arch.enums import ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import CheckpointHook, TrainData, TrainManager, VisualizeHook
from generativelib.preprocessing.preprocessor import DataPreprocessor
from generativelib.preprocessing.processors import *


class TrainContext(ABC):
    def __init__(
        self, 
        config_serializer: TrainConfigDeserializer,
        model_type: ModelTypes,
        model_template_type: Type[OptimizationTemplate]
    ) -> None:
        self.config_serializer = config_serializer
        self.model_type = model_type
        self.model_template_type = model_template_type
    
    @abstractmethod
    def _init_visualize_hook(self, template: OptimizationTemplate) -> VisualizeHook:
        pass
    
    @abstractmethod
    def _add_model_evaluators(self, template: OptimizationTemplate) -> None:
        pass
    
    def _preprocessor_metadata(self) -> Dict[str, Any]:
        paths = self.config_serializer.params_by_section(section='path', keys=['masks', 'dataset'])
        
        dataset_preprocessor = DataPreprocessor(
            *paths.values(),
            files_extensions=['.png'],
            processors=[
                RotateMask(),
                AdjustToContent(),
                Crop(k=0.5),
            ]
        )
        return dataset_preprocessor.get_metadata()
    
    def _model_template(self) -> OptimizationTemplate:
        model_params = self.config_serializer.model_params(self.model_type)
        arch_collection = self.config_serializer.optimize_collection(self.model_type)
        template = self.model_template_type(model_params, arch_collection)
        
        return template
    
    def _dataset_creator(self, dataset_metadata: Dict[str, Any], transforms: torchvision.transforms.v2.Compose) -> DatasetCreator:
        mask_processors = self.config_serializer.all_dataset_masks()
        dataset_params = self.config_serializer.get_global_section("dataset")

        return DatasetCreator(
            metadata=dataset_metadata,
            mask_processors=mask_processors,
            transforms=transforms,
            dataset_params=dataset_params
        )
    
    def _visualize_hook(
        self,
        gen_callable: Callable
    ) -> VisualizeHook:
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        visualize_interval = glob_train_params.get('visualize_interval', 5)
        
        return VisualizeHook(
            generate_fn=gen_callable,
            interval=visualize_interval
        )
    
    def _checkpoint_hook(self) -> CheckpointHook:
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        checkpoint_interval = glob_train_params.get('checkpoint_interval', 25)
        return CheckpointHook(checkpoint_interval)
    
    def _train_data(
        self,
        visualize_hook: VisualizeHook,
        checkpoint_hook: CheckpointHook
    ) -> TrainData:
        # глобальные данные обучения и путей
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        glob_path_params = self.config_serializer.get_global_section(PATH_KEY)
        
        # путь для вывода
        processed_path = glob_path_params.get('processed', '')
        model_output_path = self.model_type.name.lower()
        final_output_path = os.path.join(processed_path, model_output_path)
        
        return TrainData(
            epochs=glob_train_params.get('epochs', 1000),
            model_out_folder=final_output_path,
            visualize_hook=visualize_hook,
            checkpoint_hook=checkpoint_hook
        )
    
    def get_train_manager(self, device: torch.device) -> TrainManager:
        # метаданные о датасете
        metadata = self._preprocessor_metadata()
        
        # получение целевого размера изображения для нейросети
        img_size = self.config_serializer.params_by_section(
            section=DATASET_KEY, 
            keys='img_size'
        )
        
        # определение Compose трансформаций для GAN
        transforms = self._get_model_transform(cast(int, img_size))

        # создание шаблона обучения для GAN
        template = self._model_template()
        
        # добавление специальных лоссов, свойственных архитектуре GAN
        self._add_model_evaluators(template)
        
        # создание хука визуализации
        visualize_hook = self._init_visualize_hook(template)
        
        # создание хука для чекпоинта
        checkpoint_hook = self._checkpoint_hook()
        
        # определение данных, необходимых в TrainManager
        train_data = self._train_data(
            visualize_hook=visualize_hook,
            checkpoint_hook=checkpoint_hook
        )
        
        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        dataloaders = ds_creator.create_loaders()
        
        return TrainManager(
            device=device,
            optim_template=template, 
            train_data=train_data,
            dataloaders=dataloaders
        )
    
    @abstractmethod
    def _get_model_transform(img_size: int):
        pass