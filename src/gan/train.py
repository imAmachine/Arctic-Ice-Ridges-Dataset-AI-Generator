import os
import torch
import matplotlib

from src.common.utils import Utils
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from src.gan.model import GenerativeModel
from src.gan.dataset import DatasetCreator


class GANTrainer:
    def __init__(self, model: GenerativeModel, dataset_processor: DatasetCreator, output_path, 
                 epochs, batch_size, device, load_weights=True, val_ratio=0.2, checkpoints_ratio=5):
        self.device = device
        self.model = model
        self.dataset_processor = dataset_processor
        self.load_weights = load_weights
        
        # параметры для обучения
        self.epochs = epochs
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.checkpoints_ratio = checkpoints_ratio
        
        self.metrics_history = {'gen_losses': [], 'disc_losses': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # пути
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
    
    def _load_checkpoint(self):
        if self.load_weights:
            try:
                _ = self.model.load_checkpoint(self.output_path)
                print(f"Checkpoint загружен успешно. Продолжаем с итерации {self.model.current_iteration}.")
            except FileNotFoundError as e:
                print(f"Ошибка при загрузке чекпоинта: {e}")
                print("Начинаем обучение с нуля.")
    
    def train(self):
        loaders = self.dataset_processor.create_train_dataloaders(batch_size=self.batch_size, shuffle=True, workers=6, val_ratio=self.val_ratio)
        train_loader = loaders.get('train')
        valid_loader = loaders.get('valid')
        
        if train_loader is None:
            print('Обучение без тренировочного загрузчика данных невозможно! Остановка...')
            return
        
        if valid_loader is None:
            print('Обучение без валидации запущено')

        for epoch in range(self.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")
            self.model.switch_mode('train')
            epoch_g_losses = defaultdict(float)
            epoch_d_losses = defaultdict(float)
            
            # Обучение
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
            if valid_loader is not None:
                self.model.switch_mode('eval')
                val_g_losses = defaultdict(float)
                val_d_losses = defaultdict(float)
                val_metrics = {'accuracy': [], 'f1': [], 'iou': []}
                
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(valid_loader, desc=f"Epoch {epoch+1} Val")):
                        inputs, targets, masks = [tensor.to(self.device) for tensor in batch]
                        generated = self.model.generator(inputs, masks)
                        
                        # Вычисляем потери для валидации
                        # Здесь используем критерии из модели, но не делаем шаг оптимизации
                        w_loss = self.model.g_trainer._calc_adv_loss(generated)
                        bce = self.model.g_trainer._BCE(generated * masks, targets * masks)
                        l1 = self.model.g_trainer._L1(generated, targets)
                        total_loss = w_loss * self.model.lambda_w + bce * self.model.lambda_bce + l1 * self.model.lambda_l1
                        
                        # Добавляем в статистику валидации
                        val_g_losses['total_loss'] += total_loss.item()
                        val_g_losses['w_loss'] += w_loss.item()
                        val_g_losses['bce'] += bce.item()
                        val_g_losses['l1'] += l1.item()
                        
                        # Собираем метрики по всему валидационному набору
                        acc, f1, iou = self.compute_metrics(generated, targets, masks)
                        val_metrics['accuracy'].append(acc)
                        val_metrics['f1'].append(f1)
                        val_metrics['iou'].append(iou)
                        
                        # Визуализация первого батча валидации
                        if i == 0:
                            self._visualize_batch(inputs, generated, targets, masks, epoch, phase='val')

                # Средние потери за эпоху на валидации
                avg_val_g_loss = {k: v / len(valid_loader) for k, v in val_g_losses.items()}
                avg_g_loss = {k: v / len(train_loader) for k, v in epoch_g_losses.items()}
                avg_d_loss = {k: v / len(train_loader) for k, v in epoch_d_losses.items()}
                
                # Средние метрики по всему валидационному набору
                avg_val_acc = np.mean(val_metrics['accuracy'])
                avg_val_f1 = np.mean(val_metrics['f1'])
                avg_val_iou = np.mean(val_metrics['iou'])
                
                self.metrics_history['gen_losses'].append(avg_g_loss.get('total_loss', 0.0))
                self.metrics_history['disc_losses'].append(avg_d_loss.get('total_loss', 0.0))
                self.metrics_history.setdefault('val_gen_loss', []).append(avg_val_g_loss.get('total_loss', 0.0))
                self.metrics_history.setdefault('accuracy', []).append(avg_val_acc)
                self.metrics_history.setdefault('f1', []).append(avg_val_f1)
                self.metrics_history.setdefault('iou', []).append(avg_val_iou)
                
                print(f"\nEpoch {epoch + 1} summary:")
                print(f"  Generator losses: {avg_g_loss}")
                print(f"  Critic losses: {avg_d_loss}")
                print(f"  Validation loss: {avg_val_g_loss}")
                print(f"  Metrics: Accuracy={avg_val_acc}, F1={avg_val_f1}, IoU={avg_val_iou}")

                self.save_loss_plot()

    def _visualize_batch(self, inputs, generated, targets, masks, epoch, phase='train'):
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

    def save_loss_plot(self, suffix=''):
        """Сохраняет график истории ошибок"""
        plt.figure(figsize=(12, 5))
        plt.plot(self.metrics_history['gen_losses'], label='Generator Loss')
        plt.plot(self.metrics_history['disc_losses'], label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"training_history{f'_{suffix}' if suffix else ''}.png"
        filepath = os.path.join(self.output_path, filename)
        plt.savefig(filepath)
        plt.close()

    def compute_metrics(self, prediction, target, mask):
        """
        prediction, target, mask — torch.Tensor [B, 1, H, W] или [B, H, W]
        """
        prediction = prediction.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        prediction = (prediction > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        mask = (mask > 0.5).astype(np.uint8)

        scores = {'accuracy': [], 'f1': [], 'iou': []}

        for i in range(prediction.shape[0]):
            pred = prediction[i].flatten()
            targ = target[i].flatten()
            m = mask[i].flatten()

            # Выбираем только пиксели по маске
            p_masked = pred[m == 1]
            t_masked = targ[m == 1]

            scores['accuracy'].append(accuracy_score(t_masked, p_masked))
            scores['f1'].append(f1_score(t_masked, p_masked, zero_division=1))
            scores['iou'].append(jaccard_score(t_masked, p_masked, zero_division=1))

        # Возвращаем среднее по батчу
        return (
            np.mean(scores['accuracy']),
            np.mean(scores['f1']),
            np.mean(scores['iou'])
        )