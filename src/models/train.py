import os
from typing import Dict
from tqdm import tqdm
import matplotlib

import torch
from torch.utils.data import DataLoader

from src.common.enums import *
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
        dataloaders: Dict[ExecPhase, DataLoader],
        output_path: str,
        epochs: int,
        checkpoints_ratio: int = 15
    ):
        self.device = device
        self.model = model

        self.visualizer = Visualizer(output_path)
        self.epochs = epochs
        
        self.weight_path = output_path
        self.checkpoints_ratio = checkpoints_ratio
        
        self.dataloaders = dataloaders

    def run(self) -> None:
        for epoch_id in range(self.epochs):
            print(f"\n=== Epoch {epoch_id + 1}/{self.epochs}")
            for phase, loader in self.dataloaders.items():
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
            for phase, loader in self.dataloaders.items():
                inp, target = [el.to(self.device) for el in next(iter(loader))]
                gen = self.model.trainers[ModelType.GENERATOR].module.arch(inp)
                self.visualizer.save(inp, target, gen, phase)
            
            if (epoch_id + 1) % self.checkpoints_ratio == 0:
                self.model.checkpoint_save(os.path.join(self.weight_path, 'training_checkpoint.pt'))