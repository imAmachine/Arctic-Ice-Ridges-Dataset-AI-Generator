from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
import torch
from tqdm import tqdm
from typing import Any, Dict, Tuple, cast

# enums
from generativelib.model.arch.enums import Modules

# base
from generativelib.model.enums import ExecPhase
from generativelib.model.train.base import ModuleOptimizersCollection, OptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *

import torch
from tqdm import tqdm
from typing import Dict, Any, Tuple


class DiffusionTemplate(OptimizationTemplate):
    def __init__(
        self,
        model_params: Dict[str, Any],
        arch_optimizers: ModuleOptimizersCollection
    ):
        super().__init__(model_params, arch_optimizers)
        num_timesteps = int(model_params.get("num_timesteps", 1000))
        self.scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
        self.scheduler.set_timesteps(num_timesteps)
        self.optim = self.optimizers.by_type(Modules.DIFFUSION)

    def _make_timesteps(self, b_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,  # type: ignore
            (b_size,),
            dtype=torch.int64,
            device=device
        )

    def _add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        noisy_full = self.scheduler.add_noise(clean, noise, timesteps) # type: ignore
        return noisy_full, noise

    def _train(self, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:
        t = self._make_timesteps(trg.size(0), trg.device)
        noisy, noise = self._add_noise(trg, t)

        model_in   = (
            self.scheduler.scale_model_input(noisy, t)  # type: ignore
            if hasattr(self.scheduler, "scale_model_input")
            else noisy
        )
        noise_pred = self.optim.module(model_in, t)
        self.optim.optimize(noise_pred, noise, mask)

    def _valid(self, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:
        t = self._make_timesteps(trg.size(0), trg.device)
        noisy, noise = self._add_noise(trg, t)

        model_in = (
            self.scheduler.scale_model_input(noisy, t)  # type: ignore
            if hasattr(self.scheduler, "scale_model_input")
            else noisy
        )

        with torch.no_grad():
            noise_pred = self.optim.module(model_in, t)
            self.optim.validate(noise_pred, noise, mask)
    
    @torch.no_grad()
    def generate(self, inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.optim.mode_to(ExecPhase.VALID)
        
        orig_timesteps = self.scheduler.timesteps.clone()
        self.scheduler.set_timesteps(50, device=inp.device)
        img = inp + torch.randn_like(inp)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
            ts = torch.full((inp.size(0),), t, device=inp.device, dtype=torch.long) # type: ignore
            model_in = (self.scheduler.scale_model_input(img, ts)
                        if hasattr(self.scheduler, "scale_model_input") else img)

            noise_pred = self.optim.module(model_in, ts)
            img = self.scheduler.step(noise_pred, t, img).prev_sample # type: ignore

        self.scheduler.timesteps = orig_timesteps
        return inp, img.clamp(-1.0, 1.0)