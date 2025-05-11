import os
from typing import Dict, List, Literal
import torch
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_score, jaccard_score
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from src.gan.model import GenerativeModel
from src.gan.dataset import DatasetCreator
from src.common.analyze_tools import FractalAnalyzerGPU
from src.common.utils import Utils


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
        
        self.metrics_history = {'train': defaultdict(list), 'valid': defaultdict(list)}
        self.losses = {'train': defaultdict(list), 'valid': defaultdict(list)}
        self.loaders = None
        self.patience_counter = 0

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def train(self):
        self.loaders = self.dataset_processor.create_train_dataloaders(
            batch_size=self.batch_size, shuffle=True, workers=6, val_ratio=self.val_ratio
        )
        train_loader = self.loaders.get('train')
        valid_loader = self.loaders.get('valid')

        if train_loader is None:
            print('Обучение без тренировочного загрузчика данных невозможно! Остановка...')
            return

        if valid_loader is None:
            print('Обучение без валидации запущено')

        if self.load_weights:
            self._load_checkpoint()

        for epoch in range(self.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")

            epoch_metrics = {'train': defaultdict(list), 'valid': defaultdict(list)}

            for phase, loader in [('train', train_loader), ('valid', valid_loader)]:
                if loader is None:
                    continue

                self.model.switch_mode('train' if phase == 'train' else 'eval')

                for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {phase.capitalize()}")):
                    self.model.switch_mode(phase)
                    _ = self._process_batch(phase=phase, batch=batch, visualize_batch=(i == 0))
                    
                    self.model.switch_mode('valid')
                    metrics = self._calc_metrics(batch)
                    epoch_metrics[phase].update(metrics)

                for metric, values in epoch_metrics[phase].items():
                    self.metrics_history[phase][metric].append(np.mean(values))

            self._show_epoch_metrics(epoch_metrics)
            
            # сохранение графиков метрик
            # for metric in self.metrics_history['train']:
            #     self.save_metric_plot(target_metric=metric, suffix=metric)

            # расчёт и вывод средних лоссов по эпохе
            for trainer in [self.model.g_trainer, self.model.c_trainer]:
                print(trainer.epoch_avg_losses_str('train', len(batch)))
            
            self._schedulers_step('valid')

            if (epoch + 1) % self.checkpoints_ratio == 0 and self.checkpoints_ratio != 0:
                self.model.save_checkpoint(self.output_path)
    
    def _schedulers_step(self, phase: Literal['train', 'valid']) -> None:
        trg_metric_val = self.metrics_history[phase][self.model.optimization_params.get('metric')][-1]
        self.model.step_schedulers(trg_metric_val)
    
    def _process_batch(self, phase: Literal['train', 'valid'], batch: List, visualize_batch=False):
        inputs, targets, masks = [tensor.to(self.device) for tensor in batch]

        if phase == 'train':
            self.model.train_step(inputs, targets, masks)
        else:
            self.model.eval_step(inputs, targets, masks)

        with torch.no_grad():
            generated = self.model.generator(inputs, masks)
            if visualize_batch:
                self._visualize_batch(inputs, generated, targets, masks, phase=phase)

        return generated

    def _visualize_batch(self, inputs, generated, targets, masks, phase='train'):
        plt.figure(figsize=(20, 12), dpi=300)

        for i in range(min(3, inputs.shape[0])):
            plt.subplot(3, 4, i*4 + 1)
            plt.imshow(inputs[i].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Input {i+1}", fontsize=12, pad=10)
            plt.axis('off')

            plt.subplot(3, 4, i*4 + 2)
            plt.imshow(masks[i].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Mask_inpaint {i+1}", fontsize=12, pad=10)
            plt.axis('off')

            plt.subplot(3, 4, i*4 + 3)
            plt.imshow(generated[i].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Generated {i+1}", fontsize=12, pad=10)
            plt.axis('off')

            plt.subplot(3, 4, i*4 + 4)
            plt.imshow(targets[i].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
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
        damaged, target, damage_mask = [el.to(self.device).detach() for el in batch]
        
        with torch.no_grad():
            generated = self.model.generator(damaged, damage_mask)

            # Accuracy (по маске)
            gen_bin = (generated.cpu().numpy() > 0.15).astype(np.uint8)
            tgt_bin = (target.cpu().numpy() > 0.15).astype(np.uint8)
            mask_bin = (damage_mask.cpu().numpy() > 0.15).astype(np.uint8)

            acc, f1, iou = [], [], []
            for i in range(gen_bin.shape[0]):
                p = gen_bin[i].flatten()
                t = tgt_bin[i].flatten()
                m = mask_bin[i].flatten()

                p_masked = p[m == 1]
                t_masked = t[m == 1]

                acc.append(precision_score(t_masked, p_masked, zero_division=1))
                f1.append(f1_score(t_masked, p_masked, zero_division=1))
                iou.append(jaccard_score(t_masked, p_masked, zero_division=1))

            accuracy = acc if acc else 0.0
            # fd = self._calc_fractal_loss(generated, target)
            
            return {
                'accuracy': np.mean(accuracy),
                'f1': np.mean(f1),
                'iou': np.mean(iou),
                # 'fd': fd
            }

    def _load_checkpoint(self):
        if self.load_weights:
            try:
                _ = self.model.load_checkpoint(self.output_path)
                print(f"Checkpoint загружен успешно. Продолжаем с итерации {self.model.current_iteration}.")
            except FileNotFoundError as e:
                print(f"Ошибка при загрузке чекпоинта: {e}")
                print("Начинаем обучение с нуля.")

    def _calc_fractal_loss(self, generated: torch.Tensor, target: torch.Tensor) -> float:
        """
        Считает среднюю разницу фрактальной размерности между сгенерированным изображением и ground truth
        """
        fd_total = 0.0
        batch_size = min(generated.shape[0], 4)

        for i in range(batch_size):
            gen_img = generated[i].detach().squeeze()
            tgt_img = target[i].detach().squeeze()

            fd_gen = FractalAnalyzerGPU.calculate_fractal_dimension(
                *FractalAnalyzerGPU.box_counting(gen_img)
            )
            fd_target = FractalAnalyzerGPU.calculate_fractal_dimension(
                *FractalAnalyzerGPU.box_counting(tgt_img)
            )

            fd_total += abs(fd_gen - fd_target)

        return fd_total / batch_size