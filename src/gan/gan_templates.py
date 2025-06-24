from typing import  Dict, Tuple

import torch

# enums
from generativelib.model.arch.enums import Modules

# base
from generativelib.model.train.base import ModuleOptimizersCollection, OptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *


class GanTemplate(OptimizationTemplate):
    def __init__(self, model_params: Dict, arch_optimizers: ModuleOptimizersCollection):
        super().__init__(model_params, arch_optimizers)
        self.n_critic = model_params.get('n_critic', 5)
        self.gen_optim = self.optimizers.by_type(Modules.GAN_GENERATOR)
        self.discr_optim = self.optimizers.by_type(Modules.GAN_DISCRIMINATOR)
    
    def _add_noise(self, inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        noisy_inp = inp.clone()
        noise = torch.rand_like(inp) * 2 - 1
        noisy_inp = torch.where(mask > 0.5, noise, noisy_inp)
        return noisy_inp
    
    def _train(self, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:
        for _ in range(self.n_critic):
            with torch.no_grad():
                _, fake = self.generate(inp, mask)
            self.discr_optim.optimize(fake, trg, mask)
        
        _, fake = self.generate(inp, mask)
        self.gen_optim.optimize(fake, trg, mask)

    def _valid(self, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:        
        with torch.no_grad():
            _, fake = self.generate(inp, mask)
            for optimizer in self.optimizers:
                optimizer.validate(fake, trg, mask)

    def generate(self, inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_inp = self._add_noise(inp, mask)
        return noisy_inp, self.gen_optim.module(noisy_inp)