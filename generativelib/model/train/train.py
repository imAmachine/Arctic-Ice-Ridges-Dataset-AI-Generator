from __future__ import annotations
import os
from typing import Dict

import torch
from tqdm import tqdm

from generativelib.model.enums import ExecPhase
from generativelib.model.train.base import BaseTrainTemplate
from torch.utils.data import DataLoader


class TrainConfigurator:
    def __init__(
        self,
        device: torch.device,
        epochs: int=1000,
        checkpoint_ratio: int=25,
        visualizer_path: str='',
        weights_path: str='',
    ):
        self.device = device
        self.epochs= epochs
        self.checkpoint_ratio = checkpoint_ratio
        
        self.visualizations_path = visualizer_path
        self.weights_path = weights_path


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
                desc = phase.name
        
                print(f"\n=== Epoch {epoch_id + 1}/{epochs}")
                for inp, target in tqdm(loader, desc=desc):
                    inp, trg = inp.to(device), target.to(device)
                    self.train_strategy.step(phase, inp, trg)