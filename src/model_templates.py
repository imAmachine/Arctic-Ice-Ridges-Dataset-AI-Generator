from typing import  Dict, List
from diffusers import DDPMScheduler

import torch
from tqdm import tqdm
import torchvision.transforms.v2 as T

from generativelib.model.arch.custom_transforms import OneOf, RandomRotate, BinarizeTransform
from generativelib.model.arch.gan import GanDiscriminator, GanGenerator

# enums
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.enums import ExecPhase

# base
from generativelib.model.train.base import ArchOptimizersCollection, BaseTrainTemplate
from generativelib.model.evaluators.base import EvalItem

# evaluators
from generativelib.model.evaluators.losses import *


class DiffusionTrainTemplate(BaseTrainTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ArchOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=model_params.get('num_timesteps', 1000)
        )

        self.dif_optim = self.arch_optimizers.by_type(GenerativeModules.DIFFUSION)

    def _add_noise(self, target: torch.Tensor, timestamp: torch.Tensor) -> Dict[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(target)
        noise_image = self.scheduler.add_noise(target, noise, timestamp)
        return noise_image, noise

    def _train_step(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (inp.size(0),), device=inp.device).long()

        noisy_images, noise = self._add_noise(inp, timesteps)

        noise_pred = self.dif_optim.arch_module(noisy_images, timesteps)
        self.dif_optim.optimize(noise_pred, noise)

    def _valid_step(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (inp.size(0),), device=inp.device).long()

        noisy_images, noise = self._add_noise(inp, timesteps)
        with torch.no_grad():
            noise_pred = self.dif_optim.arch_module(noisy_images, timesteps)
            for optimizer in self.arch_optimizers:
                _ = optimizer.loss(noise_pred, noise, ExecPhase.VALID)

    def _generate_from_noise(self, target: torch.Tensor) -> torch.Tensor:
        # [AI METHOD]
        noise = torch.randn_like(target)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        timesteps = self.scheduler.timesteps
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling"):
                timesteps_tensor = t.expand(target.size(0)).to(target.device)
                noise_pred = self.dif_optim.arch_module(noise, timesteps_tensor)

                scheduler_output = self.scheduler.step(noise_pred, t, noise)
                noise = scheduler_output.prev_sample

        return noise
    
    @staticmethod
    def get_transforms(target_img_size) -> List[T.Transform]:
        max_crop = 1024
        return T.Compose([
            T.ToImage(), 
            OneOf([T.RandomCrop((size, size)) for size in range(640, max_crop, 128)], p = 1.0), 

            RandomRotate(p = 0.8), 
            T.RandomHorizontalFlip(p = 0.8), 
            T.RandomVerticalFlip(p = 0.8), 

            T.Resize((target_img_size, target_img_size), interpolation = T.InterpolationMode.BILINEAR), 
            T.ToDtype(torch.float32, scale = True), 
            BinarizeTransform(p = 1.0), 
            T.Normalize(mean=[0.5], std=[0.5]),
        ])


class GANTrainTemplate(BaseTrainTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ArchOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.n_critic = self.model_params.get("n_critic")
        
        self.gen_optim = self.arch_optimizers.by_type(GenerativeModules.GAN_GENERATOR)        
        self.discr_optim = self.arch_optimizers.by_type(GenerativeModules.GAN_DISCRIMINATOR)
        
        self.arch_optimizers.add_evals(self._default_evaluators(self.discr_optim.arch_module))
    
    def _train_step(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = self.gen_optim.arch_module(inp)
            self.discr_optim.optimize(fake, target)
        
        fake = self.gen_optim.arch_module(inp)
        self.gen_optim.optimize(fake, target)

    def _valid_step(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            fake = self.gen_optim.arch_module(inp)
            for optimizer in self.arch_optimizers:
                _ = optimizer.loss(fake, target, ExecPhase.VALID)
    
    def _default_evaluators(self, discriminator: GanDiscriminator) -> Dict[GenerativeModules, List[EvalItem]]:
        """Create dictionary of evaluation metrics and losses."""
        return {
            GenerativeModules.GAN_GENERATOR: [EvalItem(GeneratorLoss(discriminator), name=LossName.ADVERSARIAL.name, type=EvaluatorType.LOSS, weight=1.0)],
            GenerativeModules.GAN_DISCRIMINATOR: [
                EvalItem(WassersteinLoss(discriminator), name=LossName.WASSERSTEIN.name, type=EvaluatorType.LOSS, weight=1.0),
                EvalItem(GradientPenalty(discriminator), name=LossName.GRADIENT_PENALTY.name, type=EvaluatorType.LOSS, weight=10.0, exec_phase=ExecPhase.TRAIN)
            ]
        }

    @staticmethod
    def get_transforms(target_img_size) -> List[T.Transform]:
        max_crop = 1024
        return T.Compose([
            T.ToImage(), 
            OneOf([T.RandomCrop((size, size)) for size in range(640, max_crop, 128)], p = 1.0), 

            RandomRotate(p = 0.8), 
            T.RandomHorizontalFlip(p = 0.8), 
            T.RandomVerticalFlip(p = 0.8), 

            T.Resize((target_img_size, target_img_size), interpolation = T.InterpolationMode.BILINEAR), 
            T.ToDtype(torch.float32, scale = True), 
            BinarizeTransform(p = 1.0),
        ])