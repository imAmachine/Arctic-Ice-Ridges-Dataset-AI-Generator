import torch
import torchvision.transforms.v2 as T

from diffusers import DDPMScheduler
from PIL import Image
from tqdm import tqdm
from typing import cast

from generativelib.model.arch.common_transforms import get_common_transforms, get_infer_transforms
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.inference.base import ModuleInference
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

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
    def __init__(self, config_serializer: TrainConfigDeserializer, model_type: ModelTypes):
        super().__init__(config_serializer, model_type)
    
    def _model_template(self) -> DiffusionTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.DIFFUSION)
        arch_collection = self.config_serializer.optimize_collection(ModelTypes.DIFFUSION)

        template = DiffusionTemplate(model_params, arch_collection)

        return template
    
    def _add_model_evaluators(self, template: DiffusionTemplate) -> None:
        diffusion = template.get_dif_optimizer().module
        optimizers_collection = template.model_optimizers
        
        optimizers_collection.add_evals({
            GenerativeModules.DIFFUSION:
            [EvalItem(
                nn.MSELoss(diffusion), 
                name=LossName.MSE.name, 
                type=EvaluatorType.LOSS, 
                weight=1.0
            )]
        })

    def _get_model_transform(self, img_size: int) -> T.Compose:
        base_transforms = get_common_transforms(cast(int, img_size))
        transforms = T.Compose([
            base_transforms,
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        return transforms
    
    def _visualize_for_model(self, template: DiffusionTemplate) -> VisualizeHook:
        return self._visualize_hook(
            gen_callable=template._generate_from_noise
        )