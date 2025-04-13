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
                 epochs, batch_size, load_weights=True, val_ratio=0.2, checkpoints_ratio=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.load_weights = load_weights
        self.dataset_processor = dataset_processor
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics_history = {'gen_losses': [], 'disc_losses': []}
        self.start_epoch = 0
        self.checkpoints_ratio = checkpoints_ratio
        
        self.val_ratio = val_ratio
        os.makedirs(self.output_path, exist_ok=True)
        
    def train(self):
        train_loader, val_loader = self.dataset_processor.create_dataloaders(
            batch_size=self.batch_size, shuffle=True, workers=6, val_ratio=self.val_ratio
        )

        best_val_loss = float('inf')
        patience_counter = 0
        self.model.generator.train()
        
        if self.load_weights:
            try:
                loaded = self.model._load_weights(self.output_path)
                if loaded:
                    print(f"Checkpoint загружен успешно. Продолжаем с итерации {self.model.current_iteration}.")
                else:
                    print("Загружены только веса моделей. Метрики и состояние оптимизаторов сброшены.")
            except FileNotFoundError:
                print("Веса не найдены. Начинаем обучение с нуля.")

        for epoch in range(self.epochs):
            epoch_g_losses = defaultdict(float)
            epoch_d_losses = defaultdict(float)

            print(f"\nЭпоха {epoch + 1}/{self.epochs}")
            
            # Обучение
            self.model.generator.train()
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
                inputs, targets, masks = [tensor.to(self.device) for tensor in batch]
                losses = self.model.train_step(inputs, targets, masks)
                
                # Обновление потерь
                for key, value in losses['g_losses'].items():
                    epoch_g_losses[key] += value
                for key, value in losses['d_losses'].items():
                    epoch_d_losses[key] += value
                
                # Визуализация первого батча
                if i == 0:
                    with torch.no_grad():
                        generated = self.model.generator(inputs, masks)
                        self._visualize_batch(inputs, generated, targets, masks, epoch, phase='train')
            
            # Валидация
            self.model.generator.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Val")):
                    inputs, targets, masks = [tensor.to(self.device) for tensor in batch]
                    generated = self.model.generator(inputs, masks)
                    
                    # Визуализация первого батча валидации
                    if i == 0:
                        self._visualize_batch(inputs, generated, targets, masks, epoch, phase='val')

            # Средние потери за эпоху
            avg_g_loss = {k: v / len(train_loader) for k, v in epoch_g_losses.items()}
            avg_d_loss = {k: v / len(train_loader) for k, v in epoch_d_losses.items()}
            self.metrics_history['gen_losses'].append(avg_g_loss.get('total_loss', 0.0))
            self.metrics_history['disc_losses'].append(avg_d_loss.get('total_loss', 0.0))

            print(f"\nEpoch {epoch + 1} summary:")
            print(f"  Generator losses: {avg_g_loss}")
            print(f"  Critic losses: {avg_d_loss}")
            
            # Сохранение лучшей модели
            val_loss = avg_g_loss.get('total_loss')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем полный checkpoint состояния обучения
                self.model.save_checkpoint(self.output_path)
                print("  🔥 Best model updated")
            else:
                patience_counter += 1
                print(f"  No improvement epochs: {patience_counter}")
                
            # Сохраняем текущий прогресс каждые 5 эпох для возможности восстановления
            if (epoch + 1) % self.checkpoints_ratio == 0:
                checkpoint_dir = os.path.join(self.output_path, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.model.save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_epoch"))
                print(f"  Сохранен промежуточный checkpoint (эпоха {epoch+1})")

    def _visualize_batch(self, inputs, generated, targets, masks, epoch, phase='train'):
        plt.figure(figsize=(20, 12), dpi=300)
        
        for i in range(min(3, inputs.shape[0])):
            plt.subplot(3, 4, i*4 + 1)
            plt.imshow(inputs[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Input {i+1}", fontsize=12, pad=10)
            plt.axis('off')
            
            plt.subplot(3, 4, i*4 + 2)
            plt.imshow(masks[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Mask {i+1}", fontsize=12, pad=10)
            plt.axis('off')
            
            plt.subplot(3, 4, i*4 + 3)
            plt.imshow(generated[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Generated {i+1}", fontsize=12, pad=10)
            plt.axis('off')
            
            plt.subplot(3, 4, i*4 + 4)
            plt.imshow(targets[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Target {i+1}", fontsize=12, pad=10)
            plt.axis('off')
        
        plt.suptitle(f'Epoch: {epoch+1} | Phase: {phase}', fontsize=14, y=1.02)
        plt.tight_layout(pad=3.0)
        
        output_file = f"{self.output_path}/{phase}.png"
        plt.savefig(output_file)
        plt.close()