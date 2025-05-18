import os
from tqdm import tqdm
import matplotlib

import torch
from torch.utils.data import DataLoader

from src.common.enums import *
from src.dataset.loader import DatasetCreator
from src.models.base import GenerativeModel
from src.models.visualizer import Visualizer
from src.models.gan.evaluators import *


# Enable cuDNN autotuner for potential performance boost
torch.backends.cudnn.benchmark = True
matplotlib.use('Agg')


class Trainer:
    """Coordinates data loading, model steps, evaluation, and visualization."""
    def __init__(
        self,
        device: torch.device,
        model: GenerativeModel,
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
            self.model.print_epoch_evaluators(epoch_id)
            
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