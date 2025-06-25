import torch
import numpy

from diffusers import DDPMScheduler
from PIL import Image
from tqdm import tqdm

from generativelib.model.arch.enums import Modules, ModelTypes
from generativelib.model.evaluators.losses import *
from generativelib.model.inference.base import ModuleInference
from generativelib.preprocessing.processors import *

from src.config_deserializer import InferenceConfigDeserializer
from src.infer_context import InferenceContext


class DiffusionInferenceContext(InferenceContext):
    def __init__(self, config: InferenceConfigDeserializer, device: torch.device):
        super().__init__(config, device)

        params = config.get_global_section(Modules.DIFFUSION)
        self.scheduler = DDPMScheduler(num_train_timesteps=params.get("num_timesteps", 1000))

        self._load_params()
        self._load_model()

    def _load_model(self):
        arch_module = self.config.create_arch_module(
            model_type=ModelTypes.DIFFUSION,
            module_name="diffusion"
        )
        self.generator = ModuleInference(Modules.DIFFUSION, arch_module.module).to(self.device)

    def load_weights(self, path: str):
        self.generator.load_weights(path)

    def generate_from_mask(self, image: numpy.ndarray) -> Image.Image:
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