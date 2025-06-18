from typing import  Dict

import torch

# enums
from generativelib.model.arch.enums import Modules

# base
from generativelib.model.train.base import ModuleOptimizer, ModuleOptimizersCollection, OptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *


class GanTemplate(OptimizationTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ModuleOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.n_critic = model_params.get('n_critic', 5)
        self.gen_optim = self.optimizers.by_type(Modules.GAN_GENERATOR)
        self.discr_optim = self.optimizers.by_type(Modules.GAN_DISCRIMINATOR)
    
    def _train(self, inp: torch.Tensor, trg: torch.Tensor) -> None:
        if self.gen_optim is None:
            raise ValueError('Оптимизатор генератора is None')
        
        if self.discr_optim is None:
            raise ValueError('Оптимизатор дискриминатора is None')
        
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = self.gen_optim.module(inp)
            self.discr_optim.optimize(fake, trg)
        
        fake = self.gen_optim.module(inp)
        self.gen_optim.optimize(fake, trg)

    def _valid(self, inp: torch.Tensor, trg: torch.Tensor) -> None:
        if self.gen_optim is None:
            raise ValueError('Оптимизатор генератора is None')
        
        with torch.no_grad():
            fake = self.gen_optim.module(inp)
            for optimizer in self.optimizers:
                optimizer.validate(fake, trg)
