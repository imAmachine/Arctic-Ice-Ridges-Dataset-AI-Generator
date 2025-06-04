from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
from tqdm import tqdm

from generativelib.model.enums import ExecPhase
from generativelib.model.train.base import BaseHook, OptimizationTemplate
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
        train_template: OptimizationTemplate,
        train_configurator: TrainConfigurator,
        visualizer: BaseHook,
        dataloaders: Dict[ExecPhase, DataLoader],
    ):
        self.hook = visualizer
        self.dataloaders = dataloaders
        self.train_strategy = train_template
        self.train_configurator = train_configurator
    
    def run(self) -> None:
        device = self.train_configurator.device
        epochs = self.train_configurator.epochs
        
        self.train_strategy.to(device)
        arch_optimizers = self.train_strategy.arch_optimizers # коллекция архитектурных оптимизаторов (ModuleOptimizersCollection) для управления всеми оптимизаторами
        
        for epoch_id in range(epochs):
            for phase, loader in self.dataloaders.items():
                print(f"\n=== Epoch {epoch_id + 1}/{epochs} === ЭТАП: {phase.name}\n")
                
                arch_optimizers.all_mode_to(phase) # переключает режим архитектурных модулей, обнуляет историю по эпохам в модулях
                for inp, target in tqdm(loader):
                    inp, trg = inp.to(device), target.to(device)
                    self.train_strategy.step(phase, inp, trg) # Вызывает реализацию шага обучения конкретной стратегии (GAN/DIFFUSION Template...)
                
                self.hook.on_phase_end(device, epoch_id, phase, loader) # вызывает хук после окончания фазы (В ДАННЫЙ МОМЕНТ ВИЗУАЛИЗАЦИЯ)
                
                arch_optimizers.all_print_phase_summary(phase) # выводит summary за эпоху по конкретной фазе (TRAIN/VALID)