from typing import Dict
import torch
import torchvision.transforms.v2 as T

from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.model.arch.common_transforms import get_common_transforms, get_infer_transforms
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.train.base import OptimizationTemplate, ModuleOptimizersCollection
from generativelib.model.common.visualizer import Visualizer
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import CheckpointHook, TrainConfigurator, TrainManager, DiffusionVisualizeHook
from generativelib.preprocessing.processors import *
from generativelib.model.evaluators.losses import *

from src.config_deserializer import TrainConfigDeserializer
from src.create_context import TrainContext
from src.diffusion.diffusion_templates import Diffusion_OptimizationTemplate


class DiffusionTrainContext(TrainContext):
    def __init__(self, config_serializer: TrainConfigDeserializer):
        super().__init__(config_serializer)
    
    def _model_template(self) -> OptimizationTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.DIFFUSION)

        arch_collection = self.config_serializer.optimize_collection(ModelTypes.DIFFUSION)
        self._model_specific_evals(arch_collection)

        train_template = Diffusion_OptimizationTemplate(model_params, arch_collection)

        return train_template
    
    def _model_specific_evals(self, optimizers_collection: ModuleOptimizersCollection):
        diffusion = optimizers_collection.by_type(GenerativeModules.DIFFUSION).module
        
        optimizers_collection.add_evals({
            GenerativeModules.DIFFUSION:
            [EvalItem(
                nn.MSELoss(diffusion), 
                name=LossName.MSE.name, 
                type=EvaluatorType.LOSS, 
                weight=1.0
            )]
        })
    
    def _train_manager(self, train_template: Diffusion_OptimizationTemplate, 
                      train_configurator: TrainConfigurator, 
                      dataloaders: Dict[ExecPhase, Dict]) -> TrainManager:
        visualizer_path = self.config_serializer.params_by_section(section=PATH_KEY, keys=Visualizer.__class__.__name__.lower())
        visualizer = DiffusionVisualizeHook(train_template, visualizer_path, train_configurator.checkpoint_ratio)
        checkpointer = CheckpointHook(train_configurator.checkpoint_ratio, train_configurator.weights)
        
        return TrainManager(
            optim_template=train_template,
            train_configurator=train_configurator,
            visualizer=visualizer,
            checkpointer=checkpointer,
            dataloaders=dataloaders,
        )
    
    def init_train(self, device: torch.device):
        # предобработка и подгрузка метаданных
        metadata = self._preprocessor_metadata()
        img_size = self.config_serializer.params_by_section(section=DATASET_KEY, keys='img_size')

        # получение текущего шаблона для обучения
        template = self._model_template()

        base_transforms = get_common_transforms(img_size)
        transforms = T.Compose([
            base_transforms,
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        train_configurator = self._train_configurator(device)

        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        dataloaders = ds_creator.create_loaders()
        
        return self._train_manager(
            template, 
            train_configurator, 
            dataloaders
            )