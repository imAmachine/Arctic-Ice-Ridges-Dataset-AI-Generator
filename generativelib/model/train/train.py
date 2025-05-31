from __future__ import annotations
from typing import Dict

import torch

from generativelib.model.enums import ExecPhase
from generativelib.model.train.base import BaseTrainTemplate
from torch.utils.data import DataLoader


class TrainConfigurator:
    def __init__(
        self,
        device: torch.device,
        epochs: int=1000,
        checkpoint_ratio: int=25,
        vizualizations: str='',
        weights: str='',
    ):
        self.device = device
        self.epochs= epochs
        self.checkpoint_ratio = checkpoint_ratio
        
        self.visualizations_path = vizualizations
        self.weights_path = weights


class TrainManager:
    def __init__(
        self,
        train_template: BaseTrainTemplate,
        train_configurator: TrainConfigurator,
        dataloaders: Dict[ExecPhase, DataLoader],
    ):
        self.train_strategy = train_template
        self.dataloaders = dataloaders
        self.train_configurator = train_configurator
    
    def run(self) -> None:
        device = self.train_configurator.device
        epochs = self.train_configurator.epochs
        
        for epoch_id in range(epochs):
            for phase, loader in self.dataloaders.items():
                print(f"\n=== Epoch {epoch_id + 1}/{epochs}\n")
                self.train_strategy.epoch(device, phase, loader)