from typing import List
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import torch.nn as nn

from src.models.structs import ArchModule, BaseModel, EvaluatorsCollector, Visualizer
from src.common.enums import ExecPhase, ModelType
from src.dataset.dataset import DatasetCreator
from src.models.gan.gan_evaluators import *
from src.models.gan.gan_arch import WGanCritic, WGanGenerator

import os
from settings import *

# Enable cuDNN autotuner for potential performance boost
torch.backends.cudnn.benchmark = True
matplotlib.use('Agg')


class GAN(BaseModel):
    """Wasserstein GAN logic with multiple critic updates per generator update."""
    def __init__(self, device: torch.device, n_critic: int = 5):
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
    
    def _init_modules(self, config_section: dict) -> List[ArchModule]:
        """Construct ArchModule list for GAN model."""
        gen = WGanGenerator(
            input_channels=1,
            feature_maps=config_section['model_base_features']
        ).to(DEVICE)
        
        disc = WGanCritic(
            input_channels=1,
            feature_maps=config_section['model_base_features']
        ).to(DEVICE)
        
        g_optimizer = GAN._create_optimizer(gen.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
        g_scheduler = GAN._create_scheduler(g_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

        d_optimizer = GAN._create_optimizer(disc.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
        d_scheduler = GAN._create_scheduler(d_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

        evaluators = GAN._create_evaluators(disc)

        modules = [
            ArchModule(
                model_type=ModelType.GENERATOR,
                arch=gen,
                optimizer=g_optimizer,
                scheduler=g_scheduler,
                eval_funcs=evaluators,
                eval_settings=config_section['evaluators_info'][ModelType.GENERATOR.value]
            ),
            ArchModule(
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
    def _create_evaluators(discriminator: WGanCritic) -> Dict:
        """Create dictionary of evaluation metrics and losses."""
        return {
            LossName.ADVERSARIAL.value: AdversarialLoss(discriminator),
            LossName.BCE.value: nn.BCELoss(),
            LossName.L1.value: nn.L1Loss(),
            LossName.WASSERSTEIN.value: WassersteinLoss(discriminator),
            LossName.GP.value: GradientPenalty(discriminator),
            MetricName.PRECISION.value: sklearn_wrapper(precision_score, DEVICE),
            MetricName.F1.value: sklearn_wrapper(f1_score, DEVICE),
            MetricName.IOU.value: sklearn_wrapper(jaccard_score, DEVICE),
        }


class Trainer:
    """Coordinates data loading, model steps, evaluation, and visualization."""
    def __init__(
        self,
        device: torch.device,
        model: BaseModel,
        dataset: DatasetCreator,
        output_path: str,
        epochs: int,
        batch_size: int,
        val_ratio: float = 0.2,
        checkpoints_ratio: int = 15
    ):
        self.device = device
        self.model = model

        self.visualizer = Visualizer(output_path)
        self.epochs = epochs
        self.weight_path = output_path
        self.checkpoints_ratio = checkpoints_ratio
        
        self.trainer_phases = dataset.get_dataloaders(
            batch_size, shuffle=True, workers=6, val_ratio=val_ratio
        )

    def run(self) -> None:
        for epoch_id in range(self.epochs):
            print(f"\n=== Epoch {epoch_id + 1}/{self.epochs}")
            for phase, loader in self.trainer_phases.items():
                self._run_epoch(phase, loader)
                
            self.model.collect_epoch_evaluators()
            self.model.print_evaluators(epoch_id)
            
            self._after_epoch(epoch_id)
    
    def _run_epoch(self, phase: ExecPhase, loader: DataLoader) -> None:
        """Метод определяет алгоритм одной эпохи, с автоматическим подсчётом метрик"""
        desc = phase.value.capitalize()
        for inp, target in tqdm(loader, desc=desc):
            inp, target = inp.to(self.device), target.to(self.device)
            self.model.model_step(inp, target, phase)

    def _after_epoch(self, epoch_id: int):
        with torch.no_grad():
            for phase, loader in self.trainer_phases.items():
                inp, target = [el.to(self.device) for el in next(iter(loader))]
                gen = self.model.trainers[ModelType.GENERATOR].module.arch(inp)
                self.visualizer.save(inp, target, gen, phase)
            
            if epoch_id + 1 % self.checkpoints_ratio == 0:
                self.model.save(os.path.join(self.weight_path, 'training_checkpoint.pt'))