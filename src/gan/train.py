import os
from typing import Dict, List, Literal
import torch
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from src.gan.model import GenerativeModel
from src.gan.dataset import DatasetCreator
from src.common.utils import Utils
from src.common.structs import ExecPhase as phases, EvaluatorType as eval_type, ModelType as m_type

class GANTrainer:
    def __init__(self, model: GenerativeModel, dataset_processor: DatasetCreator, output_path, 
                 epochs, batch_size, device, load_weights=True, val_ratio=0.2, checkpoints_ratio=25):
        self.device = device
        self.model = model
        self.dataset_processor = dataset_processor
        self.load_weights = load_weights

        self.epochs = epochs
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.checkpoints_ratio = checkpoints_ratio
        
        self.loaders = None
        self.patience_counter = 0

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def train(self):
        self.loaders = self.dataset_processor.create_train_dataloaders(
            batch_size=self.batch_size, shuffle=True, workers=6, val_ratio=self.val_ratio
        )
        
        train_loader = self.loaders.get(phases.TRAIN)
        valid_loader = self.loaders.get(phases.VALID)

        if train_loader is None:
            print('Обучение без тренировочного загрузчика данных невозможно! Остановка...')
            return

        if valid_loader is None:
            print('Обучение без валидации запущено')

        if self.load_weights:
            self._load_checkpoint()

        for epoch in range(self.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")

            for phase, loader in [(phases.TRAIN, train_loader), (phases.VALID, valid_loader)]:
                if loader is None:
                    continue

                self.model.switch_phase(phases.TRAIN if phase == phases.TRAIN else phases.EVAL)

                for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {phase.value.capitalize()}")):
                    self.model.switch_phase(phase)
                    batch = [el.to(self.device) for el in batch]
                    
                    _ = self._process_batch(phase=phase, batch=batch, visualize_batch=(i == 0))

            # расчёт и вывод средних лоссов по эпохе
            self.model.g_trainer.get_summary(name=m_type.GENERATOR.value, phase=phases.TRAIN)
            self.model.d_trainer.get_summary(name=m_type.DISCRIMINATOR.value, phase=phases.TRAIN)
            
            if (epoch + 1) % self.checkpoints_ratio == 0 and self.checkpoints_ratio != 0:
                self.model.save_checkpoint(self.output_path)
    
    def _process_batch(self, phase: phases, batch: tuple, visualize_batch=False):
        if phase == phases.TRAIN:
            self.model.train_step(batch)
        elif phase == phases.VALID:
            self.model.valid_step(batch)

        with torch.no_grad():
            generated = self.model.generator(batch[0], batch[2])
            if visualize_batch:
                self._visualize_batch(batch, generated, phase=phase)

        return generated

    def _visualize_batch(self, batch: tuple, generated, phase: phases=phases.TRAIN):
        inputs, targets, masks = batch
        
        plt.figure(figsize=(20, 12), dpi=300)

        for i in range(min(3, inputs.shape[0])):
            plt.subplot(3, 4, i*4 + 1)
            plt.imshow(inputs[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Input {i+1}", fontsize=12, pad=10)
            plt.axis('off')

            plt.subplot(3, 4, i*4 + 2)
            plt.imshow(masks[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Mask_inpaint {i+1}", fontsize=12, pad=10)
            plt.axis('off')

            plt.subplot(3, 4, i*4 + 3)
            plt.imshow(generated[i].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Generated {i+1}", fontsize=12, pad=10)
            plt.axis('off')

            plt.subplot(3, 4, i*4 + 4)
            plt.imshow(targets[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Target {i+1}", fontsize=12, pad=10)
            plt.axis('off')

        plt.suptitle(f'Phase: {phase}', fontsize=14, y=1.02)
        plt.tight_layout(pad=3.0)

        output_file = f"{self.output_path}/{phase.value}.png"
        plt.savefig(output_file)
        plt.close()

    def save_metrics_plots(self, phase: str = phases.VALID.value):
        """Сохраняет графики для всех метрик указанной фазы"""
        phase_metrics = self.model.g_trainer.evaluate_processor.evaluators_history[phase][-1].get(eval_type.METRIC.value)
        
        for metric_name, values in phase_metrics.items():
            plt.figure(figsize=(12, 5))
            
            plt.plot(values, label=phase, color='orange')

            plt.xlabel('Epoch')
            plt.ylabel(metric_name.capitalize())
            plt.title(f'{metric_name.capitalize()} History ({phase})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # Сохранение файла
            filename = f"{metric_name}_{phase}_history.png"
            filepath = os.path.join(self.output_path, filename)
            plt.savefig(filepath)
            plt.close()
