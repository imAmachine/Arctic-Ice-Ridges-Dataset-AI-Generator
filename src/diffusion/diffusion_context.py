import torch
import torchvision.transforms.v2 as T

from diffusers import DDPMScheduler
from PIL import Image
from tqdm import tqdm
from typing import Dict

from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.model.arch.common_transforms import get_common_transforms, get_infer_transforms
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.train.base import OptimizationTemplate, ModuleOptimizersCollection
from generativelib.model.common.visualizer import Visualizer
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import CheckpointHook, TrainConfigurator, TrainManager, VisualizeHook
from generativelib.preprocessing.processors import *
from generativelib.model.evaluators.losses import *
from generativelib.model.inference.base import ModuleInference

from src.config_deserializer import TrainConfigDeserializer, InferenceConfigDeserializer
from src.create_context import TrainContext, InferenceContext
from src.diffusion.diffusion_templates import DiffusionTemplate


class DiffusionInferenceContext(InferenceContext):
    def __init__(self, config: InferenceConfigDeserializer):
        super().__init__(config)

        params = config.get_global_section(GenerativeModules.DIFFUSION)
        self.scheduler = DDPMScheduler(num_train_timesteps=params.get("num_timesteps", 1000))

        self._load_params()
        self._load_model()

    def _load_model(self):
        arch_module = self.config.create_arch_module(
            model_type=ModelTypes.DIFFUSION,
            module_name="diffusion"
        )
        self.generator = ModuleInference(GenerativeModules.DIFFUSION, arch_module.module).to(self.device)

    def load_weights(self, path: str):
        self.generator.load_weights(path)

    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        tensor = self._prepare_input_image(image)
        with torch.no_grad():
            generated = self._generate_from_noise(tensor.unsqueeze(0))
        return self._postprocess(generated, image)
    
    def _generate_from_noise(self, target: torch.Tensor) -> torch.Tensor:
        # [AI METHOD]
        noise = torch.randn_like(target)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        
        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
                timesteps_tensor = t.expand(target.size(0)).to(target.device)
                noise_pred = self.generator.generate(noise, timesteps_tensor)

                scheduler_output = self.scheduler.step(noise_pred, t, noise)
                noise = scheduler_output.prev_sample

        return noise


class DiffusionTrainContext(TrainContext):
    def __init__(self, config_serializer: TrainConfigDeserializer):
        super().__init__(config_serializer)
    
    def _model_template(self) -> OptimizationTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.DIFFUSION)

        arch_collection = self.config_serializer.optimize_collection(ModelTypes.DIFFUSION)
        self._model_specific_evals(arch_collection)

        train_template = DiffusionTemplate(model_params, arch_collection)

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
    
    def _train_manager(self, train_template: DiffusionTemplate, 
                      train_configurator: TrainConfigurator, 
                      dataloaders: Dict[ExecPhase, Dict]) -> TrainManager:
        visualizer_path = self.config_serializer.params_by_section(section=PATH_KEY, keys=Visualizer.__class__.__name__.lower())
        
        visualizer = VisualizeHook(
            callable_fn=train_template._generate_from_noise, 
            output_path=visualizer_path, 
            interval=train_configurator.checkpoint_ratio
        )
        
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