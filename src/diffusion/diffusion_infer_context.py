import torch
import numpy

from diffusers import DDPMScheduler
from PIL import Image
from tqdm import tqdm
from typing import Tuple

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
        inp, mask = self._prepare_input_image(image)
        with torch.no_grad():
            generated = self._generate_from_noise(inp.unsqueeze(0), mask.unsqueeze(0))
        return self._postprocess(generated, image)
    
    def _add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        noisy_full = self.scheduler.add_noise(clean, noise, timesteps) # type: ignore
        return noisy_full, noise
    
    def _generate_from_noise(self, inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_timesteps = self.scheduler.timesteps.clone()
        self.scheduler.set_timesteps(50, device=inp.device)
        img = torch.randn_like(inp)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
            ts = torch.full((inp.size(0),), t, device=inp.device, dtype=torch.long) # type: ignore
            model_in = (self.scheduler.scale_model_input(img, ts)
                        if hasattr(self.scheduler, "scale_model_input") else img)

            noise_pred = self.generator.generate(model_in, ts)
            img = self.scheduler.step(noise_pred, t, img).prev_sample # type: ignore

            noised_target, _ = self._add_noise(inp, ts)
            img[~mask.bool()] = noised_target[~mask.bool()]

        self.scheduler.timesteps = orig_timesteps

        return img.clamp(-1.0, 1.0)