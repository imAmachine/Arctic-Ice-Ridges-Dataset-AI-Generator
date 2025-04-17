import os
from typing import Dict, List, Literal
import torch
import matplotlib

from src.common.utils import Utils
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from src.gan.model import GenerativeModel
from src.gan.dataset import DatasetCreator


class GANTrainer:
    def __init__(self, model: GenerativeModel, dataset_processor: DatasetCreator, output_path, 
                 epochs, batch_size, device, load_weights=True, val_ratio=0.2, checkpoints_ratio=5):
        self.device = device
        self.model = model
        self.dataset_processor = dataset_processor
        self.load_weights = load_weights

        self.epochs = epochs
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.checkpoints_ratio = checkpoints_ratio

        self.metrics_history = {'train': defaultdict(list), 'valid': defaultdict(list)}
        self.losses = {'train': defaultdict(list), 'valid': defaultdict(list)}
        self.patience_counter = 0

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def train(self):
        loaders = self.dataset_processor.create_train_dataloaders(
            batch_size=self.batch_size, shuffle=True, workers=6, val_ratio=self.val_ratio
        )
        train_loader = loaders.get('train')
        valid_loader = loaders.get('valid')

        if train_loader is None:
            print('Обучение без тренировочного загрузчика данных невозможно! Остановка...')
            return

        if valid_loader is None:
            print('Обучение без валидации запущено')

        if self.load_weights:
            self._load_checkpoint()

        for epoch in range(self.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")

            epoch_metrics = {
                'train': {'precision': [], 'f1': [], 'iou': []},
                'valid': {'precision': [], 'f1': [], 'iou': []}
            }

            for phase, loader in [('train', train_loader), ('valid', valid_loader)]:
                if loader is None:
                    continue

                self.model.switch_mode('train' if phase == 'train' else 'eval')

                for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {phase.capitalize()}")):
                    self._process_batch(mode=phase, batch=batch, visualize_batch=(i == 0))
                    acc, f1, iou = self._calc_metrics(batch)
                    epoch_metrics[phase]['precision'].append(acc)
                    epoch_metrics[phase]['f1'].append(f1)
                    epoch_metrics[phase]['iou'].append(iou)

                self.metrics_history[phase]['precision'].append(np.mean(epoch_metrics[phase]['precision']))
                self.metrics_history[phase]['f1'].append(np.mean(epoch_metrics[phase]['f1']))
                self.metrics_history[phase]['iou'].append(np.mean(epoch_metrics[phase]['iou']))

            self._show_epoch_metrics(epoch_metrics)
            self.save_metric_plot(target_metric='iou')

            if (epoch + 1) % self.checkpoints_ratio == 0:
                self.model.save_checkpoint(self.output_path)

    def _process_batch(self, mode: Literal['train', 'valid'], batch: List, visualize_batch=False):
        inputs, targets, masks = [tensor.to(self.device) for tensor in batch]

        if mode == 'train':
            self.model.train_step(inputs, targets, masks)
        else:
            self.model.eval_step(inputs, targets, masks)

        if visualize_batch:
            with torch.no_grad():
                generated = self.model.generator(inputs, masks)
                self._visualize_batch(inputs, generated, targets, masks, phase=mode)

    def _visualize_batch(self, inputs, generated, targets, masks, phase='train'):
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

        plt.suptitle(f'Phase: {phase}', fontsize=14, y=1.02)
        plt.tight_layout(pad=3.0)

        output_file = f"{self.output_path}/{phase}.png"
        plt.savefig(output_file)
        plt.close()

    def save_metric_plot(self, target_metric, suffix=''):
        plt.figure(figsize=(12, 5))
        plt.plot(self.metrics_history['train'][target_metric], label='train')
        plt.plot(self.metrics_history['valid'][target_metric], label='valid')
        plt.xlabel('Epoch')
        plt.ylabel(target_metric)
        plt.title('Train history')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"training_history{f'_{suffix}' if suffix else ''}.png"
        filepath = os.path.join(self.output_path, filename)
        plt.savefig(filepath)
        plt.close()

    def _show_epoch_metrics(self, epoch_metrics: Dict):
        avg_metrics = {'train': defaultdict(list), 'valid': defaultdict(list)}

        for mode_name, metrics in epoch_metrics.items():
            for metric_name, metric_history in metrics.items():
                avg_metrics[mode_name].update({metric_name: np.mean(metric_history)})

        output_str = ''
        for mode, metrics in avg_metrics.items():
            output_str += f'[{mode} metrics]: '
            for metric_name, val in metrics.items():
                output_str += f'{metric_name} - {val:.2f} '
            output_str += '\n'

        print(output_str)

    def _calc_metrics(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float, float]:
        damaged, original, damage_mask = [el.to(self.device) for el in batch]
        generated = self.model.generator(damaged, damage_mask)

        generated_uint8 = (generated.detach().cpu().numpy() > 0).astype(np.uint8)
        original_uint8 = (original.detach().cpu().numpy() > 0).astype(np.uint8)
        damage_mask_uint8 = (damage_mask.detach().cpu().numpy() > 0).astype(np.uint8)

        acc, f1, iou = [], [], []
        for i in range(damaged.size(0)):
            pred = generated_uint8[i].flatten()
            targ = original_uint8[i].flatten()
            m = damage_mask_uint8[i].flatten()

            p_masked = pred[m == 1]
            t_masked = targ[m == 1]

            acc.append(precision_score(t_masked, p_masked, zero_division=1))
            f1.append(f1_score(t_masked, p_masked, zero_division=1))
            iou.append(jaccard_score(t_masked, p_masked, zero_division=1))

        return np.mean(acc), np.mean(f1), np.mean(iou)

    def _load_checkpoint(self):
        if self.load_weights:
            try:
                _ = self.model.load_checkpoint(self.output_path)
                print(f"Checkpoint загружен успешно. Продолжаем с итерации {self.model.current_iteration}.")
            except FileNotFoundError as e:
                print(f"Ошибка при загрузке чекпоинта: {e}")
                print("Начинаем обучение с нуля.")
