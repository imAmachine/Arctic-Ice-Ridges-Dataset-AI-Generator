from collections import defaultdict
import os
from typing import Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.analyzer.fractal_funcs import FractalAnalyzerGPU
from src.gan.model import GenerativeModel
from src.datasets.dataset import DatasetCreator
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GANTrainer:
    def __init__(self, model: GenerativeModel, dataset_processor: DatasetCreator, output_path, 
                 epochs=10, batch_size=10, load_weights=True, early_stop_patience=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.load_weights = load_weights
        self.dataset_processor = dataset_processor
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        
        # Инициализация планировщиков скорости обучения
        # self.g_scheduler = ReduceLROnPlateau(self.model.g_trainer.optimizer, mode='max', factor=0.5, patience=6, verbose=True)
        # self.d_scheduler = ReduceLROnPlateau(self.model.d_trainer.optimizer, mode='max', factor=0.5, patience=6, verbose=True)
        
        self.epoch_g_losses = defaultdict(float)
        self.epoch_d_losses = defaultdict(float)
        
        # Для ранней остановки
        self.best_val_psnr = -np.inf
        self.epochs_no_improve = 0
        
        os.makedirs(self.output_path, exist_ok=True)
        self._init_metrics()

    def _init_metrics(self):
        # Сбрасываем метрики при начале новой эпохи
        self.epoch_g_losses.clear()
        self.epoch_d_losses.clear()
        self.metric_history = {
            'train': {'ssim': [], 'binary_accuracy': [], 'iou': [], 
                    'direction_similarity': [], 'fractal_dimension_diff': [],
                    'fractal_gen': [], 'fractal_target': []},
            'val': {'ssim': [], 'binary_accuracy': [], 'iou': [], 
                    'direction_similarity': [], 'fractal_dimension_diff': [],
                    'fractal_gen': [], 'fractal_target': []}
        }

    def calculate_metrics(self, generated, target, mask):
        generated_np = generated.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        
        batch_size = generated_np.shape[0]
        metrics = {
            'ssim': 0.0, 'binary_accuracy': 0.0, 'iou': 0.0,
            'direction_similarity': 0.0, 'fractal_dimension_diff': 0.0,
            'fractal_gen': 0.0, 'fractal_target': 0.0
        }
        
        for i in range(batch_size):
            gen_img = generated_np[i, 0]
            target_img = target_np[i, 0]
            mask_img = mask_np[i, 0]
                        
            # SSIM
            metrics['ssim'] += structural_similarity(gen_img, target_img, data_range=1.0)
            
            # Binary Accuracy и IoU
            binary_gen = (gen_img > 0.5)
            binary_target = (target_img > 0.5)
            
            correct_pixels = np.sum((binary_gen == binary_target) * mask_img)
            total_pixels = np.sum(mask_img)
            metrics['binary_accuracy'] += correct_pixels / (total_pixels + 1e-8)
            
            intersection = np.sum((binary_gen * binary_target) * mask_img)
            union = np.sum(((binary_gen + binary_target) > 0) * mask_img)
            metrics['iou'] += intersection / (union + 1e-8)
            
            gen_grad_y, gen_grad_x = np.gradient(gen_img)
            target_grad_y, target_grad_x = np.gradient(target_img)

            gen_dir = np.arctan2(gen_grad_y, gen_grad_x)
            target_dir = np.arctan2(target_grad_y, target_grad_x)

            # Учет значимых градиентов (magnitude > threshold)
            mag_gen = np.sqrt(gen_grad_x**2 + gen_grad_y**2)
            mag_target = np.sqrt(target_grad_x**2 + target_grad_y**2)
            significant = (mag_gen > 0.05) & (mag_target > 0.05) & (mask_img > 0.5)

            if np.any(significant):
                angle_diff = np.abs(gen_dir - target_dir)
                angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
                metrics['direction_similarity'] += 1 - np.mean(angle_diff[significant]/np.pi)
            else:
                metrics['direction_similarity'] += 0.0  # Нет значимых градиентов
            
            # Анализ фрактальной размерности
            # try:
            #     gen_tensor = torch.tensor(gen_img, device=self.device).float()
            #     target_tensor = torch.tensor(target_img, device=self.device).float()
                
            #     fd_gen = FractalAnalyzerGPU.calculate_fractal_dimension(*FractalAnalyzerGPU.box_counting(gen_tensor))
            #     fd_target = FractalAnalyzerGPU.calculate_fractal_dimension(*FractalAnalyzerGPU.box_counting(target_tensor))
                
            #     metrics['fractal_dimension_diff'] += abs(fd_gen - fd_target)
            #     metrics['fractal_gen'] += fd_gen
            #     metrics['fractal_target'] += fd_target
            # except Exception as e:
            #     print(f"Ошибка расчета фракталов: {e}")

        for key in metrics:
            metrics[key] /= batch_size
            
        return metrics

    def _visualize_batch(self, inputs, generated, targets, epoch, phase='train'):
        plt.figure(figsize=(15, 6))
        for i in range(min(3, inputs.shape[0])):
            plt.subplot(3, 3, i+1)
            plt.imshow(inputs[i].cpu().squeeze(), cmap='gray')
            plt.title(f"Input {i+1}")
            
            plt.subplot(3, 3, i+4)
            plt.imshow(generated[i].cpu().squeeze(), cmap='gray')
            plt.title(f"Generated {i+1}")
            
            plt.subplot(3, 3, i+7)
            plt.imshow(targets[i].cpu().squeeze(), cmap='gray')
            plt.title(f"Target {i+1}")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/{phase}_epoch.png")
        plt.close()

    def _print_metrics(self, metrics, phase='Train'):
        table = [
            ["SSIM", f"{metrics['ssim']:.4f}"],
            ["IoU", f"{metrics['iou']:.4f}"],
            ["Binary Accuracy", f"{metrics['binary_accuracy']:.4f}"],
            ["Direction Similarity", f"{metrics['direction_similarity']:.4f}"],
            ["Fractal Difference", f"{metrics['fractal_dimension_diff']:.4f}"],
            ["Fractal (Gen)", f"{metrics['fractal_gen']:.4f}"],
            ["Fractal (Target)", f"{metrics['fractal_target']:.4f}"]
        ]
        print(f"\n{phase} Metrics:")
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    def train(self):
        train_loader, val_loader = self.dataset_processor.create_dataloaders(batch_size=self.batch_size, shuffle=True, workers=6)
        
        for epoch in range(self.epochs):
            self.model.generator.train()
            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch)
            val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch)
            
            # Обновление планировщиков
            # self.g_scheduler.step(val_metrics['iou'])
            # self.d_scheduler.step(self.epoch_d_losses['binary_accuracy'])
            
            if val_metrics['direction_similarity'] > self.best_val_psnr:
                self.best_val_psnr = val_metrics['direction_similarity']
                self.model._save_models(f"{self.output_path}")

        self._plot_metrics()
        return self.metric_history

    def _run_epoch(self, loader, training=True, epoch=0):
        self._init_metrics()
        metrics = {'ssim': 0.0, 'binary_accuracy': 0.0, 
                 'iou': 0.0, 'direction_similarity': 0.0, 
                 'fractal_dimension_diff': 0.0, 'fractal_gen': 0.0, 'fractal_target': 0.0}
        
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {'Train' if training else 'Val'}")):
            data = self._process_batch(batch, training)
            
            if not training and i == 0:
                self._visualize_batch(data['inputs'], data['generated'], data['targets'], epoch, 
                                   phase='train' if training else 'val')
            
            for key in metrics:
                metrics[key] += data['metrics'][key]
                
        for key in metrics:
            metrics[key] /= len(loader)
            phase = 'train' if training else 'val'
            self.metric_history[phase][key].append(metrics[key])
            
        self._print_metrics(metrics, 'Train' if training else 'Val')
        return metrics

    def _process_batch(self, batch, training):
        inputs, targets, masks = [tensor.to(self.device) for tensor in batch]
        
        if training:
            losses = self.model.train_step(inputs, targets, masks)
            self._update_losses(losses)
        
        with torch.set_grad_enabled(training):
            generated = self.model.generator(inputs, masks)
            metrics = self.calculate_metrics(generated, targets, masks)
            
        return {'inputs': inputs, 'generated': generated, 'targets': targets, 'metrics': metrics}

    def _update_losses(self, losses):
        for key in losses['g_losses']:
            self.epoch_g_losses[key] = self.epoch_g_losses.get(key, 0.0) + losses['g_losses'][key]
        for key in losses['d_losses']:
            self.epoch_d_losses[key] = self.epoch_d_losses.get(key, 0.0) + losses['d_losses'][key]

    def _plot_metrics(self):
        plt.figure(figsize=(15, 10))
        
        # График потерь
        plt.subplot(2, 2, 1)
        
        # Извлекаем значения потерь из истории
        g_losses = [x.get('total_loss', 0) for x in self.model.g_trainer.loss_history]
        d_losses = [x.get('total_loss', 0) for x in self.model.d_trainer.loss_history]
        
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.title('Training Losses')
        plt.legend()
                
        plt.subplot(2, 2, 3)
        plt.plot(self.metric_history['train']['ssim'], label='Train')
        plt.plot(self.metric_history['val']['ssim'], label='Val')
        plt.title('SSIM Comparison')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.metric_history['train']['iou'], label='Train')
        plt.plot(self.metric_history['val']['iou'], label='Val')
        plt.title('IoU Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/training_metrics.png")
        plt.close()