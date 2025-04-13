import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from src.gan.model import GenerativeModel
from src.datasets.dataset import DatasetCreator


class GANTrainer:
    def __init__(self, model: GenerativeModel, dataset_processor: DatasetCreator, output_path, 
                 epochs=10, batch_size=10, load_weights=True, early_stop_patience=50):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.load_weights = load_weights
        self.dataset_processor = dataset_processor
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        
        self.epoch_g_losses = defaultdict(float)
        self.epoch_d_losses = defaultdict(float)
        
        # История метрик для отслеживания
        self.metrics_history = {
            'gen_losses': [],
            'disc_losses': []
        }
        
        os.makedirs(self.output_path, exist_ok=True)
        
    def train(self):
        train_loader, val_loader = self.dataset_processor.create_dataloaders(
            batch_size=self.batch_size, shuffle=True, workers=4
        )

        best_val_loss = float('inf')
        patience_counter = 0

        self.model.generator.train()
        if self.load_weights:
            try:
                self.model._load_weights(self.output_path)
                print("Веса модели загружены успешно.")
            except FileNotFoundError:
                print("Веса не найдены. Начинаем обучение с нуля.")

        for epoch in range(self.epochs):
            # Сброс накопленных потерь
            self.epoch_g_losses = defaultdict(float)
            self.epoch_d_losses = defaultdict(float)

            # Обучение
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")
            self._epoch_run(train_loader, training=True, epoch=epoch)
            
            # Валидация
            with torch.no_grad():
                self._epoch_run(val_loader, training=False, epoch=epoch)

            # Средние потери за эпоху
            avg_g_loss = {k: v / len(train_loader) for k, v in self.epoch_g_losses.items()}
            avg_d_loss = {k: v / len(train_loader) for k, v in self.epoch_d_losses.items()}
            self.metrics_history['gen_losses'].append(avg_g_loss.get('total_loss', 0.0))
            self.metrics_history['disc_losses'].append(avg_d_loss.get('total_loss', 0.0))

            print(f"\nEpoch {epoch + 1} summary:")
            print(f"  Generator losses: {avg_g_loss}")
            print(f"  Critic losses: {avg_d_loss}")
            
            val_loss = avg_g_loss.get('total_loss')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.model._save_models(self.output_path)
                print("  🔥 Best model updated")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{self.early_stop_patience}")

    def _epoch_run(self, loader, training=True, epoch=0):
        phase = 'Train' if training else 'Val'
        
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {phase}")):
            batch_data = self._process_batch(batch, training)
            
            if not training and i == 0:
                self._visualize_batch(batch_data['inputs'], batch_data['generated'], batch_data['targets'], batch_data['masks'], epoch, 
                                     phase='train' if training else 'val')

    def _process_batch(self, batch, training):
        inputs, targets, masks = [tensor.to(self.device) for tensor in batch]
        
        if training:
            losses = self.model.train_step(inputs, targets, masks)
            self._update_losses(losses)
        
        with torch.no_grad():
            generated = self.model.generator(inputs, masks)
            
        return {
            'inputs': inputs, 
            'generated': generated, 
            'targets': targets,
            'masks': masks
        }
    
    def _update_losses(self, losses):
        for key in losses['g_losses']:
            self.epoch_g_losses[key] = self.epoch_g_losses.get(key, 0.0) + losses['g_losses'][key]
        for key in losses['d_losses']:
            self.epoch_d_losses[key] = self.epoch_d_losses.get(key, 0.0) + losses['d_losses'][key]

    def _visualize_batch(self, inputs, generated, targets, masks, epoch, phase='train'):
        # Увеличиваем размер фигуры и DPI для высокого качества
        plt.figure(figsize=(20, 12), dpi=300)
        
        # Для каждого из первых 3 примеров в батче
        for i in range(min(3, inputs.shape[0])):
            # Input image
            plt.subplot(3, 4, i*4 + 1)
            plt.imshow(inputs[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Input {i+1}", fontsize=12, pad=10)
            plt.axis('off')
            
            # Mask image (генерированная маска)
            plt.subplot(3, 4, i*4 + 2)
            plt.imshow(masks[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Mask {i+1}", fontsize=12, pad=10)
            plt.axis('off')
            
            # Generated image
            plt.subplot(3, 4, i*4 + 3)
            plt.imshow(generated[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Generated {i+1}", fontsize=12, pad=10)
            plt.axis('off')
            
            # Target image
            plt.subplot(3, 4, i*4 + 4)
            plt.imshow(targets[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Target {i+1}", fontsize=12, pad=10)
            plt.axis('off')
        
        # Общие настройки
        plt.suptitle(f'Epoch: {epoch+1} | Phase: {phase}', fontsize=14, y=1.02)
        plt.tight_layout(pad=3.0)  # Увеличение отступов
        
        # Сохраняем изображение
        output_file = f"{self.output_path}/{phase}.png"
        plt.savefig(output_file)
        plt.close()
