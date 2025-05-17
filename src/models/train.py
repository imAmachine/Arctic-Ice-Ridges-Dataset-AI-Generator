from typing import List
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import torch.nn as nn

from src.models.structs import ArchModule, BaseModel, MetricsCollector, Visualizer
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
    def __init__(self, device: torch.device, modules: List[ArchModule], n_critic: int = 5):
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

        super().__init__(device, modules, checkpoint_map)
        self.n_critic = n_critic

    def train_step(self, inp: Tensor, target: Tensor) -> None:
        gen_mgr = self.trainers[ModelType.GENERATOR]
        disc_mgr = self.trainers[ModelType.DISCRIMINATOR]
        
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake = gen_mgr.module.arch(inp)
            disc_mgr.optimization_step(target, fake)
        
        fake = gen_mgr.module.arch(inp)
        gen_mgr.optimization_step(target, fake)

    def valid_step(self, inp: Tensor, target: Tensor) -> None:
        with torch.no_grad():
            fake = self(inp)
            for mgr in self.trainers.values():
                mgr.valid_step(target, fake)

    @staticmethod
    def build_modules(config_section: dict) -> List[ArchModule]:
        """Construct ArchModule list for GAN model."""
        # Model initialization
        gen = WGanGenerator(
            input_channels=1,
            feature_maps=config_section['model_base_features']
        ).to(DEVICE)
        disc = WGanCritic(
            input_channels=1,
            feature_maps=config_section['model_base_features']
        ).to(DEVICE)
        
        g_optimizer = GAN.create_optimizer(gen.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
        g_scheduler = GAN.create_scheduler(g_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

        d_optimizer = GAN.create_optimizer(disc.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
        d_scheduler = GAN.create_scheduler(d_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

        evaluators = GAN.build_evaluators(disc)

        modules: List[ArchModule] = []
        modules.extend([
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
        ])
        
        return modules
    
    @staticmethod
    def create_optimizer(parameters, lr: float=0.0001, betas=(0.0, 0.9)):
        """Create Adam optimizer with specified parameters."""
        return torch.optim.Adam(parameters, lr=lr, betas=betas)

    @staticmethod
    def create_scheduler(optimizer, mode: str, factor: float=0.5, patience: int=6):
        """Create ReduceLROnPlateau scheduler."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )

    @staticmethod
    def build_evaluators(discriminator: WGanCritic) -> Dict:
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
    ):
        self.device = device
        self.model = model
        self.metrics = MetricsCollector(self.model.trainers)
        self.visualizer = Visualizer(output_path)
        self.epochs = epochs
        self.weight_path = output_path
        self.metrics_history = []
        self.losses_history = []
        
        self.train_loader, self.val_loader = dataset.get_dataloaders(
            batch_size, shuffle=True, workers=6, val_ratio=val_ratio
        )

    def run(self) -> None:
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.epochs}")
            for phase, loader in ((ExecPhase.TRAIN, self.train_loader), (ExecPhase.VALID, self.val_loader)):
                self._run_epoch(phase, loader)
            
            for phase, loader in ((ExecPhase.TRAIN, self.train_loader), (ExecPhase.VALID, self.val_loader)):
                self._after_epoch(phase, loader)

            self._capture_epoch_data()
            
            self.metrics.reset()

            self.model.save(os.path.join(self.weight_path, 'training_checkpoint.pt'))

    def _capture_epoch_data(self):
        """Собирает данные из self.metrics."""
        df = self.metrics.summary(ExecPhase.VALID)
        
        # Преобразуем MultiIndex в строки
        df.index = df.index.map(lambda x: '_'.join(x))
        epoch_metrics = df.to_dict(orient='index')
        self.metrics_history.append(epoch_metrics)
        
        # Потери по шагам
        epoch_losses = {}
        for model_type, mgr in self.metrics.managers.items():
            history = mgr.evaluate_processor.evaluators_history[ExecPhase.VALID]
            serialized = []
            for step in history:
                step_clean = {}
                for eval_type, values in step.items():
                    step_clean[eval_type] = {k: float(v) for k, v in values.items()}
                serialized.append(step_clean)
            epoch_losses[model_type.value] = serialized
        
        self.losses_history.append(epoch_losses)

    def _run_epoch(self, phase: ExecPhase, loader: DataLoader) -> None:         
        desc = phase.value.capitalize()
        for inp, target in tqdm(loader, desc=desc):
            inp, target = inp.to(self.device), target.to(self.device)
            
            if phase is ExecPhase.TRAIN:
                self.model.train_step(inp, target)
            else:
                self.model.valid_step(inp, target)

    def _after_epoch(self, phase: ExecPhase, loader: DataLoader):
        with torch.no_grad():
            print(f"--- [{phase.name}] Metrics")
            self.metrics.print(phase)
            inp, target = [el.to(self.device) for el in next(iter(loader))]
            
            gen = self.model.trainers[ModelType.GENERATOR].module.arch(inp)
            self.visualizer.save(inp, target, gen, phase)