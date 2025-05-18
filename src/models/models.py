import torch
from torch import device, Tensor, nn
from typing import Dict, List

from src.common.enums import LossName, MetricName, ModelType
from src.models.base import Architecture, GenerativeModel

from src.models.gan.architecture import CustomDiscriminator, CustomGenerator
from src.models.gan.evaluators import *


class GAN(GenerativeModel):
    """Wasserstein GAN logic with multiple critic updates per generator update."""
    def __init__(self, device: device, n_critic: int = 5):
        checkpoint_map = {
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

        super().__init__(device, checkpoint_map)
        self.n_critic = n_critic

    def _train_step(self, inp: Tensor, target: Tensor) -> None:
        gen_mgr = self.trainers[ModelType.GENERATOR]
        disc_mgr = self.trainers[ModelType.DISCRIMINATOR]
        
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = gen_mgr.module.arch(inp)
            disc_mgr.optimization_step(target, fake)
        
        fake = gen_mgr.module.arch(inp)
        gen_mgr.optimization_step(target, fake)

    def _valid_step(self, inp: Tensor, target: Tensor) -> None:
        with torch.no_grad():
            fake = self(inp)
            for mgr in self.trainers.values():
                mgr.valid_step(target, fake)
    
    def _init_modules(self, config_section: dict) -> List[Architecture]:
        """Construct ArchModule list for GAN model."""
        gen = CustomGenerator(
            input_channels=1,
            feature_maps=config_section['model_base_features']
        ).to(self.device)
        
        disc = CustomDiscriminator(
            input_channels=1,
            feature_maps=config_section['model_base_features']
        ).to(self.device)
        
        g_optimizer = GAN._create_optimizer(gen.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
        g_scheduler = GAN._create_scheduler(g_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

        d_optimizer = GAN._create_optimizer(disc.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
        d_scheduler = GAN._create_scheduler(d_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

        evaluators = GAN._create_evaluators(self.device, disc)

        modules = [
            Architecture(
                model_type=ModelType.GENERATOR,
                arch=gen,
                optimizer=g_optimizer,
                scheduler=g_scheduler,
                eval_funcs=evaluators,
                eval_settings=config_section['evaluators_info'][ModelType.GENERATOR.value]
            ),
            Architecture(
                model_type=ModelType.DISCRIMINATOR,
                arch=disc,
                optimizer=d_optimizer,
                scheduler=d_scheduler,
                eval_funcs=evaluators,
                eval_settings=config_section['evaluators_info'][ModelType.DISCRIMINATOR.value]
            )
        ]
        
        return modules
    
    @staticmethod
    def _create_optimizer(parameters, lr: float=0.0001, betas=(0.0, 0.9)):
        """Create Adam optimizer with specified parameters."""
        return torch.optim.Adam(parameters, lr=lr, betas=betas)

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
    def _create_evaluators(device: torch.device, discriminator: CustomDiscriminator) -> Dict:
        """Create dictionary of evaluation metrics and losses."""
        return {
            LossName.ADVERSARIAL.value: AdversarialLoss(discriminator),
            LossName.BCE.value: nn.BCELoss(),
            LossName.L1.value: nn.L1Loss(),
            LossName.WASSERSTEIN.value: WassersteinLoss(discriminator),
            LossName.GP.value: GradientPenalty(discriminator),
            MetricName.PRECISION.value: sklearn_wrapper(precision_score, device),
            MetricName.F1.value: sklearn_wrapper(f1_score, device),
            MetricName.IOU.value: sklearn_wrapper(jaccard_score, device),
        }
