from typing import Dict, List

import torch
import torchvision.transforms.v2 as T

from generativelib.model.arch.custom_transforms import OneOf, RandomRotate
from generativelib.model.arch.gan import CustomDiscriminator, CustomGenerator

# enums
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.enums import ExecPhase

# base
from generativelib.model.arch.base import Architecture
from generativelib.model.train.base import GenerativeModel
from generativelib.model.evaluators.base import Evaluator

# evaluators
from generativelib.model.evaluators.losses import *


class GAN(GenerativeModel):
    """Wasserstein GAN logic with multiple critic updates per generator update."""
    def __init__(self, device: torch.device, n_critic: int = 5, checkpoint_map: Dict=None):
        checkpoint_map_final = checkpoint_map if checkpoint_map is not None else {
            GenerativeModules.GENERATOR: {
                'model': ('trainers', GenerativeModules.GENERATOR, 'module', 'arch'),
                'optimizer': ('trainers', GenerativeModules.GENERATOR, 'module', 'optimizer'),
                'scheduler': ('trainers', GenerativeModules.GENERATOR, 'module', 'scheduler'),
            },
            GenerativeModules.DISCRIMINATOR: {
                'model': ('trainers', GenerativeModules.DISCRIMINATOR, 'module', 'arch'),
                'optimizer': ('trainers', GenerativeModules.DISCRIMINATOR, 'module', 'optimizer'),
                'scheduler': ('trainers', GenerativeModules.DISCRIMINATOR, 'module', 'scheduler'),
            }
        }

        super().__init__(device, checkpoint_map_final)
        self.n_critic = n_critic
        self.evaluators = {
            GenerativeModules.GENERATOR.value: [],
            GenerativeModules.DISCRIMINATOR.value: []
        }
    
    def _train_step(self, inp: Tensor, target: Tensor) -> None:
        gen_mgr = self.trainers[GenerativeModules.GENERATOR]
        disc_mgr = self.trainers[GenerativeModules.DISCRIMINATOR]
        
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = gen_mgr.module(inp)
            disc_mgr.optimization_step(fake, target)
        
        fake = gen_mgr.module(inp)
        gen_mgr.optimization_step(fake, target)

    def _valid_step(self, inp: Tensor, target: Tensor) -> None:
        with torch.no_grad():
            fake = self(inp)
            for mgr in self.trainers.values():
                mgr.valid_step(fake, target)
    
    def _init_modules(self, config_section: dict) -> List[Architecture]:
        """Construct ArchModule list for GAN model."""
        base_f = config_section['model_base_features']
        optim_params = config_section['optimization_params']
        optim_betas = (0.0, 0.9)
        
        gen = CustomGenerator(
            in_ch=1,
            f_base=base_f
        ).to(self.device)
        
        disc = CustomDiscriminator(
            in_ch=1,
            f_base=base_f
        ).to(self.device)
        
        g_optimizer = GAN._create_optimizer(gen.parameters(), optim_params['lr'], betas=optim_betas)
        g_scheduler = GAN._create_scheduler(g_optimizer, optim_params['mode'], factor=0.5, patience=6)
        d_optimizer = GAN._create_optimizer(disc.parameters(), optim_params['lr'], betas=optim_betas)
        d_scheduler = GAN._create_scheduler(d_optimizer, optim_params['mode'], factor=0.5, patience=6)
        
        self._init_model_evaluators(disc)
        
        modules = [
            Architecture(
                model_type=GenerativeModules.GENERATOR,
                arch=gen,
                optimizer=g_optimizer,
                scheduler=g_scheduler,
                evaluators=self.evaluators[GenerativeModules.GENERATOR.value]
            ),
            Architecture(
                model_type=GenerativeModules.DISCRIMINATOR,
                arch=disc,
                optimizer=d_optimizer,
                scheduler=d_scheduler,
                evaluators=self.evaluators[GenerativeModules.DISCRIMINATOR.value]
            )
        ]
        
        return modules
    
    def _init_model_evaluators(self, discriminator: CustomDiscriminator) -> None:
        """Create dictionary of evaluation metrics and losses."""
        self.evaluators[GenerativeModules.GENERATOR.value].extend([Evaluator(GeneratorLoss(discriminator), name=LossName.ADVERSARIAL.value, type=EvaluatorType.LOSS.value, weight=1.0)])
        
        self.evaluators[GenerativeModules.DISCRIMINATOR.value].extend([
            Evaluator(WassersteinLoss(discriminator), name=LossName.WASSERSTEIN.value, type=EvaluatorType.LOSS.value, weight=1.0),
            Evaluator(GradientPenalty(discriminator), name=LossName.GP.value, type=EvaluatorType.LOSS.value, weight=10.0, exec_phase=ExecPhase.TRAIN.value)
        ])
    
    @staticmethod
    def _create_optimizer(parameters, lr: float=1e-4, betas=(0.0, 0.9), eps: float=1e-8):
        """Create Adam optimizer with specified parameters."""
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps)

    @staticmethod
    def _create_scheduler(optimizer, mode: str, factor: float=0.5, patience: int=6):
        """Create ReduceLROnPlateau scheduler."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    
    @staticmethod
    def get_transforms(target_img_size) -> List[T.Transform]:
        max_crop = 1024
        return [
            T.ToImage(), 
            T.Resize((max_crop, max_crop), interpolation = T.InterpolationMode.BILINEAR), 
            OneOf([T.RandomCrop((size, size)) for size in range(640, max_crop, 128)], p = 1.0), 

            RandomRotate(p = 0.8), 
            T.RandomHorizontalFlip(p = 0.8), 
            T.RandomVerticalFlip(p = 0.8), 

            T.Resize((target_img_size, target_img_size), interpolation = T.InterpolationMode.BILINEAR), 
            T.ToDtype(torch.float32, scale = True), 
        ]