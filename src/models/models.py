import torch
from torch import device, Tensor
from typing import Dict, List

from src.common.enums import EvaluatorType, ExecPhase, LossName, ModelType
from src.models.base import Architecture, Evaluator, GenerativeModel

from src.models.gan.architecture import CustomDiscriminator, CustomGenerator
from src.models.gan.evaluators import *


class GAN(GenerativeModel):
    """Wasserstein GAN logic with multiple critic updates per generator update."""
    def __init__(self, device: device, n_critic: int = 5, checkpoint_map: Dict=None):
        checkpoint_map_final = checkpoint_map if checkpoint_map is not None else {
            ModelType.GENERATOR: {
                'model': ('trainers', ModelType.GENERATOR, 'module', 'arch'),
                'optimizer': ('trainers', ModelType.GENERATOR, 'module', 'optimizer'),
                'scheduler': ('trainers', ModelType.GENERATOR, 'module', 'scheduler'),
            },
            ModelType.DISCRIMINATOR: {
                'model': ('trainers', ModelType.DISCRIMINATOR, 'module', 'arch'),
                'optimizer': ('trainers', ModelType.DISCRIMINATOR, 'module', 'optimizer'),
                'scheduler': ('trainers', ModelType.DISCRIMINATOR, 'module', 'scheduler'),
            }
        }

        super().__init__(device, checkpoint_map_final)
        self.n_critic = n_critic
        self.evaluators = {
            ModelType.GENERATOR.value: [],
            ModelType.DISCRIMINATOR.value: []
        }

    def _train_step(self, inp: Tensor, target: Tensor) -> None:
        gen_mgr = self.trainers[ModelType.GENERATOR]
        disc_mgr = self.trainers[ModelType.DISCRIMINATOR]
        
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
                model_type=ModelType.GENERATOR,
                arch=gen,
                optimizer=g_optimizer,
                scheduler=g_scheduler,
                evaluators=self.evaluators[ModelType.GENERATOR.value]
            ),
            Architecture(
                model_type=ModelType.DISCRIMINATOR,
                arch=disc,
                optimizer=d_optimizer,
                scheduler=d_scheduler,
                evaluators=self.evaluators[ModelType.DISCRIMINATOR.value]
            )
        ]
        
        return modules
    
    def _init_model_evaluators(self, discriminator: CustomDiscriminator) -> None:
        """Create dictionary of evaluation metrics and losses."""
        self.evaluators[ModelType.GENERATOR.value].extend([Evaluator(AdversarialLoss(discriminator), name=LossName.ADVERSARIAL.value, type=EvaluatorType.LOSS.value, weight=1.0)])
        
        self.evaluators[ModelType.DISCRIMINATOR.value].extend([
            Evaluator(WassersteinLoss(discriminator), name=LossName.WASSERSTEIN.value, type=EvaluatorType.LOSS.value, weight=1.0),
            Evaluator(GradientPenalty(discriminator), name=LossName.GP.value, type=EvaluatorType.LOSS.value, weight=1.0, exec_phase=ExecPhase.TRAIN.value)
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
