from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
import torch
from tqdm import tqdm
from typing import Any, Dict, Tuple, cast

# enums
from generativelib.model.arch.enums import Modules

# base
from generativelib.model.train.base import ModuleOptimizer, ModuleOptimizersCollection, OptimizationTemplate

# evaluators
from generativelib.model.evaluators.losses import *


class DiffusionTemplate(OptimizationTemplate):
    scheduler: DDPMScheduler
    optim: ModuleOptimizer

    def __init__(
        self,
        model_params: Dict[str, Any],
        arch_optimizers: ModuleOptimizersCollection
    ) -> None:
        super().__init__(model_params, arch_optimizers)

        num_ts = int(model_params.get('num_timesteps', 1000))
        self.scheduler = DDPMScheduler(num_train_timesteps=num_ts)
        self.optim = self.optimizers.by_type(Modules.DIFFUSION)

    def _make_timesteps(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Возвращает симметричный тензор timesteps dtype=int64 на нужном device.
        """
        half = batch_size // 2 + 1
        num_ts = int(self.scheduler.config.num_train_timesteps) # type: ignore

        t1 = torch.randint(
            low=0,
            high=num_ts,
            size=(half,),
            device=device,
            dtype=torch.int64
        )
        t2 = num_ts - t1 - 1
        all_ts = torch.cat([t1, t2], dim=0)[:batch_size]
        return all_ts

    def _add_noise(
        self,
        target: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Добавляем гауссов шум в target на заданных timesteps.
        """
        noise = torch.randn_like(target)
        noisy = self.scheduler.add_noise(target, noise, timesteps) # type: ignore
        return noisy, noise

    def _train(
        self,
        inp: torch.Tensor,
        trg: torch.Tensor
    ) -> None:
        timesteps = self._make_timesteps(inp.size(0), inp.device)
        noisy, noise = self._add_noise(inp, timesteps)
        noise_pred = self.optim.module(noisy, timesteps)
        self.optim.optimize(noise_pred, noise)

    def _valid(
        self,
        inp: torch.Tensor,
        trg: torch.Tensor
    ) -> None:
        timesteps = self._make_timesteps(inp.size(0), inp.device)
        noisy, noise = self._add_noise(inp, timesteps)
        with torch.no_grad():
            noise_pred = self.optim.module(noisy, timesteps)
            self.optim.validate(noise_pred, noise)

    def _generate_from_noise(
        self,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Сэмплинг из шума обратно в изображение.
        """
        noise = torch.randn_like(target)
        num_ts = int(self.scheduler.config.num_train_timesteps) # type: ignore
        self.scheduler.set_timesteps(num_ts)

        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
                step = int(t)
                ts = torch.full(
                    (target.size(0),),
                    step,
                    dtype=torch.int64,
                    device=target.device
                )
                pred = self.optim.module(noise, ts)
                
                raw_out = self.scheduler.step(pred, step, noise, return_dict=True)
                out = cast(DDPMSchedulerOutput, raw_out)
                
                noise = out.prev_sample

        return noise
