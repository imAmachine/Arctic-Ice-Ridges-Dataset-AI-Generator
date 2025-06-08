from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from tqdm import tqdm

from generativelib.model.arch.base import ArchModule
from generativelib.model.common.checkpoint import CheckpointManager
from generativelib.model.common.interfaces import ITorchState
from generativelib.model.common.visualizer import Visualizer
from generativelib.model.enums import ExecPhase
from torch.utils.data import DataLoader

from generativelib.model.train.base import OptimizationTemplate

# Enable cuDNN autotuner for potential performance boost
torch.backends.cudnn.benchmark = True

@dataclass
class TrainConfigurator:
    device: torch.device
    epochs: int=1000
    checkpoint_ratio: int=25
    vizualizations: str=''
    weights: str=''


class VisualizeHook:
    def __init__(self, generate_fn: Callable, output_path: str, interval: int):
        self.interval = interval
        self.gen = generate_fn
        self.visualizer = Visualizer(output_path)
        
    def __call__(self, device, epoch_id, phase, loader):
        if (epoch_id + 1) % self.interval == 0:
            with torch.no_grad():
                inp, trg = next(iter(loader))
                generated = self.gen(inp.to(device))
                self.visualizer.save(inp, trg, generated, phase)


class CheckpointHook:
    def __init__(self, interval: int, output_path: str):
        self.interval = interval
        self.output_path = output_path
        
    def __call__(self, epoch_id: int, obj: ITorchState):
        if (epoch_id + 1) % self.interval == 0:
            CheckpointManager.save_state(obj, self.output_path)


class TrainManager:
    def __init__(
        self,
        optim_template: OptimizationTemplate,
        train_configurator: TrainConfigurator,
        visualizer: VisualizeHook,
        checkpointer: CheckpointHook,
        dataloaders: Dict[ExecPhase, DataLoader],
    ):
        self.visualizer = visualizer
        self.checkpoint = checkpointer
        
        self.dataloaders = dataloaders
        self.optim_template = optim_template
        self.train_configurator = train_configurator
    
    def run(self, is_load_weights=False) -> None:
        device = self.train_configurator.device
        epochs = self.train_configurator.epochs
        self.optim_template.to(device) # установка device для модулей
        
        if is_load_weights:
            CheckpointManager.load_state(self.optim_template.model_optimizers, self.train_configurator.weights)
            
        
        for epoch_id in range(epochs):
            for phase, loader in self.dataloaders.items():
                print(f"\n=== Epoch {epoch_id + 1}/{epochs} === ЭТАП: {phase.name}\n")
                
                self.optim_template.mode_to(phase) # переключает режим архитектурных модулей, обнуляет историю по эпохам в модулях
                for inp, target in tqdm(loader):
                    inp, trg = inp.to(device), target.to(device)
                    self.optim_template.step(phase, inp, trg) # Вызывает реализацию шага обучения конкретной стратегии (GAN/DIFFUSION Template...)
                
                self.visualizer(device, epoch_id, phase, loader) # вызывает визуализацию батча по окончанию фазы
                self.optim_template.all_print_phase_summary(phase) # выводит summary за эпоху по конкретной фазе (TRAIN/VALID)
            
            self.checkpoint(epoch_id, self.optim_template.model_optimizers) # вызывает сохранение чекпоинта
                