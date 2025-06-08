from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict

import os
from typing import Callable, Dict

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from generativelib.model.common.checkpoint import CheckpointManager
from generativelib.model.common.interfaces import ITorchState
from generativelib.model.enums import ExecPhase
from torch.utils.data import DataLoader

from src.diffusion.diffusion_templates import Diffusion_OptimizationTemplate
from generativelib.model.train.base import OptimizationTemplate

# Enable cuDNN autotuner for potential performance boost
torch.backends.cudnn.benchmark = True

@dataclass
class TrainData:
    epochs: int
    model_out_folder: str
    visualize_hook: VisualizeHook
    checkpoint_hook: CheckpointHook


class VisualizeHook:
    def __init__(self, generate_fn: Callable, interval: int):
        self.interval = interval
        self.gen = generate_fn
    
    def __save(
        self,
        folder_path: str,
        inp: torch.Tensor, 
        trg: torch.Tensor, 
        gen: torch.Tensor, 
        phase: ExecPhase, 
        samples: int = 3
    ) -> None:
        cols = min(samples, inp.size(0), 5)
        plt.figure(figsize=(12, 12), dpi=300)
        
        for row_idx, batch in enumerate((inp, gen, trg)):
            for col_idx in range(cols):
                img = batch[col_idx].cpu().squeeze().numpy()
                ax = plt.subplot(3, cols, row_idx * cols + col_idx + 1)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"{['Input', 'Gen', 'Target'][row_idx]} {col_idx+1}")
                ax.axis('off')
        plt.suptitle(f"Phase: {phase.name}", y=1.02)
        plt.tight_layout(pad=3)

        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, f"{phase.name}.png")
        
        plt.savefig(path)
        plt.close()
    
    def __call__(self, device, folder_path: str, epoch_id, phase, loader):      
        if (epoch_id + 1) % self.interval == 0:
            with torch.no_grad():
                inp, trg = next(iter(loader))
                generated = self.callable_fn(inp.to(device))
                self.visualizer.save(inp, trg, generated, phase)


class CheckpointHook:
    def __init__(self, interval: int):
        self.interval = interval
        
    def __call__(self, folder_path: str, epoch_id: int, obj: ITorchState):
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, 'checkpoint.pt')
        
        if (epoch_id + 1) % self.interval == 0:
            CheckpointManager.save_state(obj, file_path)


class TrainManager:
    def __init__(
        self,
        device: torch.device,
        optim_template: OptimizationTemplate,
        train_data: TrainData,
        dataloaders: Dict[ExecPhase, DataLoader],
    ):
        self.device = device
        self.dataloaders = dataloaders
        self.optim_template = optim_template
        self.train_data = train_data
    
    def run(self, is_load_weights: bool=False) -> None:
        epochs = self.train_data.epochs
        
        # пути
        model_folder_path = self.train_data.model_out_folder
        checkpoint_folder = os.path.join(model_folder_path, 'checkpoints')
        visualizations_folder = os.path.join(model_folder_path, 'visualizations')
        
        # хуки
        checkpoint = self.train_data.checkpoint_hook
        visualize = self.train_data.visualize_hook
        
        # установка device для модулей
        self.optim_template.to(self.device)
        
        if is_load_weights:
            CheckpointManager.load_state(self.optim_template.model_optimizers, checkpoint_folder)
        
        for epoch_id in range(epochs):
            for phase, loader in self.dataloaders.items():
                print(f"\n=== Epoch {epoch_id + 1}/{epochs} === ЭТАП: {phase.name}\n")
                
                self.optim_template.mode_to(phase) # переключает режим архитектурных модулей, обнуляет историю по эпохам в модулях
                for inp, target in tqdm(loader):
                    inp, trg = inp.to(self.device), target.to(self.device)
                    self.optim_template.step(phase, inp, trg) # Вызывает реализацию шага обучения конкретной стратегии (GAN/DIFFUSION Template...)
                
                visualize(self.device, visualizations_folder, epoch_id, phase, loader) # вызывает визуализацию батча по окончанию фазы
                self.optim_template.all_print_phase_summary(phase) # выводит summary за эпоху по конкретной фазе (TRAIN/VALID)
            
            checkpoint(checkpoint_folder, epoch_id, self.optim_template.model_optimizers) # вызывает сохранение чекпоинта
