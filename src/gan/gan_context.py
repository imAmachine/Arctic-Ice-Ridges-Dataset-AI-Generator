from typing import Dict
from PIL import Image
import torch

from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate, ModuleOptimizersCollection
from generativelib.model.common.visualizer import Visualizer
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import CheckpointHook, TrainConfigurator, TrainManager, VisualizeHook
from generativelib.preprocessing.processors import *
from generativelib.model.inference.base import ModuleInference

from src.create_context import TrainContext, InferenceContext
from src.config_deserializer import TrainConfigDeserializer, InferenceConfigDeserializer
from src.gan.gan_templates import GanTemplate


class GanInferenceContext(InferenceContext):
    def __init__(self, config: InferenceConfigDeserializer):
        super().__init__(config)

        self._load_params()
        self._load_model()

    def _load_model(self):
        arch_module = self.config.create_arch_module(
            model_type=ModelTypes.GAN,
            module_name="gan_generator"
        )
        self.generator = ModuleInference(GenerativeModules.GAN_GENERATOR, arch_module.module).to(self.device)

    def load_weights(self, path: str):
        self.generator.load_weights(path)

    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        tensor = self._prepare_input_image(image)
        with torch.no_grad():
            generated = self.generator.generate(tensor.unsqueeze(0))
        return self._postprocess(generated, image)


# ВРЕМЕННОЕ (видимо постоянное) РЕШЕНИЕ, НУЖНО РАЗГРЕБАТЬ!!!!!!!!!!!!!!!
class GanTrainContext(TrainContext):
    def __init__(self, config_serializer: TrainConfigDeserializer):
        super().__init__(config_serializer)
    
    def _model_template(self) -> OptimizationTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.GAN)
        
        arch_collection = self.config_serializer.optimize_collection(ModelTypes.GAN)
        self._model_specific_evals(arch_collection)
        
        train_template = GanTemplate(model_params, arch_collection)
        
        return train_template
    
    # Временное решение WIP
    def _model_specific_evals(self, optimizers_collection: ModuleOptimizersCollection):
        discriminator = optimizers_collection.by_type(GenerativeModules.GAN_DISCRIMINATOR).module
        
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
    
    def _train_manager(self, train_template: GanTemplate, train_configurator: TrainConfigurator, dataloaders: Dict[ExecPhase, Dict]) -> TrainManager:
        # ВРЕМЕННОЕ (видимо постоянное) РЕШЕНИЕ
        generator = train_template.gen_optim.module
        visualizer_path = self.config_serializer.params_by_section(section=PATH_KEY, keys=Visualizer.__class__.__name__.lower())
        visualizer = VisualizeHook(generator, visualizer_path, train_configurator.checkpoint_ratio)
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
        
        transforms = get_common_transforms(img_size)
        train_configurator = self._train_configurator(device)
        
        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        dataloaders = ds_creator.create_loaders()
        
        return self._train_manager(
            template,
            train_configurator,
            dataloaders
        )