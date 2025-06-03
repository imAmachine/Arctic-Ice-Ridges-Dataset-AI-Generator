from typing import Dict, List
import torch
from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY, WEIGHT_KEY
from generativelib.dataset.base import BaseMaskProcessor
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import ArchModule, ArchOptimizersCollection, BaseHook
from src.config_wrappers import TrainConfigSerializer
from src.gan.gan_templates import GAN_OptimizationTemplate
from generativelib.dataset.loader import DatasetCreator, DatasetMaskingProcessor
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import TrainConfigurator, TrainManager
from generativelib.preprocessing.preprocessor import DataPreprocessor
from generativelib.preprocessing.processors import *
from src.visualizer import Visualizer


class VisualizeHook(BaseHook):
    def __init__(self, device: torch.device, generator: ArchModule, output_path: str, interval: int):
        super().__init__(interval)
        self.visualizer = Visualizer(output_path)
        self.generator = generator
        self.device = device
        
    def on_phase_end(self, epoch_id, phase, loader):
        if (epoch_id + 1) % self.interval == 0:
            with torch.no_grad():
                inp, trg = next(iter(loader))
                generated = self.generator(inp.to(self.device))
                
                self.visualizer.save(inp, trg, generated, phase)


# ВРЕМЕННОЕ РЕШЕНИЕ [WIP] НУЖНО РАЗГРЕБАТЬ И УНИФИЦИРОВАТЬ!!!!!!!!!!!!!!!
class GanTrainContext:
    def __init__(self, config_serializer: TrainConfigSerializer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config_serializer = config_serializer
    
    def _preprocessor_metadata(self) -> Dict:
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
    
    def _model_template(self) -> tuple:
        arch_collection = self.config_serializer.serialize_optimize_collection(self.device, ModelTypes.GAN)
        model_params = self.config_serializer.get_model_params(ModelTypes.GAN)
        
        self._model_specific_evals(arch_collection)
        
        train_template = GAN_OptimizationTemplate(model_params, arch_collection)
        
        return train_template
    
    # Временное решение WIP
    def _model_specific_evals(self, optimizers_collection: ArchOptimizersCollection):
        discriminator = optimizers_collection.by_type(GenerativeModules.GAN_DISCRIMINATOR).arch
        
        optimizers_collection.add_evals({
            GenerativeModules.GAN_GENERATOR:
            [EvalItem(
                GeneratorLoss(discriminator), 
                name=LossName.ADVERSARIAL.name, 
                type=EvaluatorType.LOSS, 
                weight=1.0
            )],
            GenerativeModules.GAN_DISCRIMINATOR:
            [
                EvalItem(
                    WassersteinLoss(discriminator), 
                    name=LossName.WASSERSTEIN.name, 
                    type=EvaluatorType.LOSS, 
                    weight=1.0
                ),
                EvalItem(
                    GradientPenalty(discriminator), 
                    name=LossName.GRADIENT_PENALTY.name, 
                    type=EvaluatorType.LOSS, 
                    weight=10.0, 
                    exec_phase=ExecPhase.TRAIN
                )
            ]
        })        
    
    def _dataset_creator(self, dataset_metadata: Dict, transforms) -> DatasetCreator:
        mask_processors: List[BaseMaskProcessor] = self.config_serializer.serialize_all_masks()
        dataset_params = self.config_serializer.get_global_section("dataset")
        
        return DatasetCreator(
            metadata=dataset_metadata,
            mask_processors=mask_processors,
            transforms=transforms,
            dataset_params=dataset_params
        )
    
    def _train_configurator(self):
        train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        
        return TrainConfigurator(
            device=self.device, 
            **train_params,
            weights=self.config_serializer.params_by_section(section=PATH_KEY, keys=WEIGHT_KEY)
        )
    
    def _train_manager(self, train_template: GAN_OptimizationTemplate, train_configurator: TrainConfigurator, dataloaders: Dict[ExecPhase, Dict]) -> TrainManager:
        
        # ВРЕМЕННОЕ РЕШЕНИЕ
        generator = train_template.gen_optim.arch
        visualizer_path = self.config_serializer.params_by_section(section=PATH_KEY, keys=Visualizer.__class__.__name__.lower())
        interval = 5
        hook = VisualizeHook(self.device, generator, visualizer_path, interval)
        
        return TrainManager(
            train_template=train_template,
            train_configurator=train_configurator,
            visualizer=hook,
            dataloaders=dataloaders,
        )
    
    def init_train(self):
        # предобработка и подгрузка метаданных
        metadata = self._preprocessor_metadata()
        img_size = self.config_serializer.params_by_section(section=DATASET_KEY, keys='img_size')
        
        # получение текущего шаблона для обучения
        template = self._model_template()
        transforms = GAN_OptimizationTemplate.get_transforms(img_size)
        train_configurator = self._train_configurator()
        
        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        dataloaders = ds_creator.create_loaders()
        
        return self._train_manager(
            template,
            train_configurator,
            dataloaders
        )