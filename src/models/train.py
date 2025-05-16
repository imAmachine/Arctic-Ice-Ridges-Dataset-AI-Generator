from typing import List
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

from src.models.structs import ArchModule, BaseModel, MetricsCollector, Visualizer
from src.common.enums import ExecPhase, ModelType
from src.dataset.dataset import DatasetCreator

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
            
            self.metrics.reset()

            self.model.save(os.path.join(self.weight_path, 'training_checkpoint.pt'))

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