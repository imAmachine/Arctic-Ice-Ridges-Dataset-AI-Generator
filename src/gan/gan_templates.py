from typing import  Dict, List

import torch
import torchvision.transforms.v2 as T

from generativelib.model.arch.custom_transforms import OneOf, RandomRotate

# enums
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.enums import ExecPhase

# base
from generativelib.model.train.base import ArchOptimizersCollection, BaseOptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *


class GAN_OptimizationTemplate(BaseOptimizationTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ArchOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.n_critic = model_params.get('n_critic', 5)
        self.gen_optim = self.arch_optimizers.by_type(GenerativeModules.GAN_GENERATOR)        
        self.discr_optim = self.arch_optimizers.by_type(GenerativeModules.GAN_DISCRIMINATOR)        
    
    def _train(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = self.gen_optim.arch_module(inp)
            self.discr_optim.optimize(fake, target)
        
        fake = self.gen_optim.arch_module(inp)
        self.gen_optim.optimize(fake, target)

    def _valid(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            fake = self.gen_optim.arch_module(inp)
            for optimizer in self.arch_optimizers:
                _ = optimizer.loss(fake, target, ExecPhase.VALID)

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
        ])