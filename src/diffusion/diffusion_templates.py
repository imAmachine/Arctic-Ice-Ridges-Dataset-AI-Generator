from diffusers import DDPMScheduler
import torch
from tqdm import tqdm
from typing import Dict, Tuple

# enums
from generativelib.model.arch.enums import GenerativeModules

# base
from generativelib.model.train.base import ModuleOptimizersCollection, OptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *

class Diffusion_OptimizationTemplate(OptimizationTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ModuleOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=model_params.get('num_timesteps', 1000)
        )
        self.dif_optim = self.arch_optimizers.by_type(GenerativeModules.DIFFUSION)

    def _add_noise(self, target: torch.Tensor, timestamp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(target)
        noise_image = self.scheduler.add_noise(target, noise, timestamp)
        return noise_image, noise

    def _train(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (inp.size(0),), device=inp.device).long()

        noisy_images, noise = self._add_noise(inp, timesteps)
        noise_fake = self.dif_optim.module(noisy_images, timesteps)
        self.dif_optim.optimize(noise_fake, noise)

    def _valid(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (inp.size(0),), device=inp.device).long()

        noisy_images, noise = self._add_noise(inp, timesteps)
        with torch.no_grad():
            noise_pred = self.dif_optim.module(noisy_images, timesteps)
            self.dif_optim.validate(noise_pred, noise)

    def _generate_from_noise(self, target: torch.Tensor) -> torch.Tensor:
        # [AI METHOD]
        noise = torch.randn_like(target)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        
        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
                timesteps_tensor = t.expand(target.size(0)).to(target.device)
                noise_pred = self.dif_optim.module(noise, timesteps_tensor)

                scheduler_output = self.scheduler.step(noise_pred, t, noise)
                noise = scheduler_output.prev_sample

        return noise