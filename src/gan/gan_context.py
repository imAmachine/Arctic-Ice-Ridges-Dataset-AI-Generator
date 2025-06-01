from typing import Dict
import torch
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import Arch, ArchOptimizersCollection, BaseHook
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
    def __init__(self, device: torch.device, generator: Arch, output_path: str, interval: int):
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
    
    def _model_template(self, model_type: str, img_size: int) -> tuple:
        if ModelTypes[model_type.upper()] == ModelTypes.GAN:
            transforms = GAN_OptimizationTemplate.get_transforms(img_size)
            arch_collection, model_params = self.config_serializer.serialize_model(self.device, ModelTypes.GAN)
            
            self._model_specific_evals(arch_collection)
            
            train_template = GAN_OptimizationTemplate(model_params, arch_collection)
        
        return train_template, transforms
    
    # Временное решение WIP
    def _model_specific_evals(self, optimizers_collection: ArchOptimizersCollection):
        discriminator = optimizers_collection.by_type(GenerativeModules.GAN_DISCRIMINATOR).arch_module
        
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
        dataset_params = self.config_serializer.get_global_section("dataset")
        masking_processor = DatasetMaskingProcessor(self.config_serializer.serialize_mask_processors())
        
        return DatasetCreator(
            metadata=dataset_metadata,
            mask_processor=masking_processor,
            transforms=transforms,
            dataset_params=dataset_params
        )
    
    def _train_manager(self, train_template: GAN_OptimizationTemplate, dataloaders: Dict[ExecPhase, Dict]) -> TrainManager:
        train_configurator = TrainConfigurator(
            device=self.device, 
            **self.config_serializer.get_global_section('train'),
            **self.config_serializer.params_by_section(section='path', keys=['vizualizations', 'weights'])
        )
        
        # ВРЕМЕННОЕ РЕШЕНИЕ
        generator = train_template.gen_optim.arch_module
        visualizer_path = train_configurator.visualizations_path
        interval = 5
        hook = VisualizeHook(self.device, generator, visualizer_path, interval)
        
        return TrainManager(
            train_template=train_template,
            train_configurator=train_configurator,
            visualizer=hook,
            dataloaders=dataloaders,
        )
    
    def init_train(self, model_type: str):
        # предобработка и подгрузка метаданных
        metadata = self._preprocessor_metadata()
        
        # получение текущего шаблона для обучения
        template, transforms = self._model_template(
            model_type, 
            self.config_serializer.params_by_section(section="arch", keys='img_size')
        )
        
        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        
        return self._train_manager(
            template, 
            ds_creator.create_loaders()
        )