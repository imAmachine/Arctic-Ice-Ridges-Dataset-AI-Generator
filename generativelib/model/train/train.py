from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import torch
from tqdm import tqdm

from generativelib.model.enums import ExecPhase
from generativelib.model.train.base import BaseHook, BaseOptimizationTemplate
from torch.utils.data import DataLoader

# Enable cuDNN autotuner for potential performance boost
torch.backends.cudnn.benchmark = True

@dataclass
class TrainConfigurator:
    device: torch.device
    epochs: int=1000
    checkpoint_ratio: int=25
    vizualizations: str=''
    weights: str=''


class TrainManager:
    def __init__(
        self,
        train_template: BaseOptimizationTemplate,
        train_configurator: TrainConfigurator,
        visualizer: BaseHook,
        dataloaders: Dict[ExecPhase, DataLoader],
    ):
        self.train_strategy = train_template
        self.hook = visualizer
        self.dataloaders = dataloaders
        self.train_configurator = train_configurator
    
    def run(self) -> None:
        device = self.train_configurator.device
        epochs = self.train_configurator.epochs
        arch_optimizers = self.train_strategy.arch_optimizers
        
        for epoch_id in range(epochs):
            for phase, loader in self.dataloaders.items():
                print(f"\n=== Epoch {epoch_id + 1}/{epochs} === ЭТАП: {phase.name}\n")
                
                arch_optimizers.all_mode_to(phase)
                for inp, target in tqdm(loader):
                    inp, trg = inp.to(device), target.to(device)
                    self.train_strategy.step(phase, inp, trg)
                
                self.hook.on_phase_end(epoch_id, phase, loader)
                
                arch_optimizers.all_print_phase_summary(phase)