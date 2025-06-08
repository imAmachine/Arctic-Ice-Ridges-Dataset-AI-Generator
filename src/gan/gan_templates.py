from typing import  Dict

import torch

# enums
from generativelib.model.arch.enums import GenerativeModules

# base
from generativelib.model.train.base import ModuleOptimizersCollection, OptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *


class GAN_OptimizationTemplate(OptimizationTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ModuleOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.n_critic = model_params.get('n_critic', 5)
        self.gen_optim = self.model_optimizers.by_type(GenerativeModules.GAN_GENERATOR)        
        self.discr_optim = self.model_optimizers.by_type(GenerativeModules.GAN_DISCRIMINATOR)        
    
    def _train(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = self.gen_optim.module(inp)
            self.discr_optim.optimize(fake, target)
        
        fake = self.gen_optim.module(inp)
        self.gen_optim.optimize(fake, target)

    def _valid(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            fake = self.gen_optim.module(inp)
            for optimizer in self.model_optimizers:
                optimizer.validate(fake, target)
