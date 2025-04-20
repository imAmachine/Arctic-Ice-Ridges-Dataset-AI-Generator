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
import torch.nn.functional as F
from scipy.linalg import sqrtm
import torchvision.models as models

from src.gan.model import GenerativeModel
from src.gan.dataset import DatasetCreator
from src.common.analyze_tools import FractalAnalyzerGPU


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
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        for param in self.inception_model.parameters():
            param.requires_grad_(False)

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
                'train': {'precision': [], 'fd': [], 'FID': [], 'f1': [], 'iou': []},
                'valid': {'precision': [], 'fd': [], 'FID': [], 'f1': [], 'iou': []}
            }

            for phase, loader in [('train', train_loader), ('valid', valid_loader)]:
                if loader is None:
                    continue

                self.model.switch_mode('train' if phase == 'train' else 'eval')
                all_real, all_fake = [], []

                for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {phase.capitalize()}")):
                    generated = self._process_batch(mode=phase, batch=batch, visualize_batch=(i == 0))
                    mask = batch[2].cpu()
                    real_imgs = batch[1].cpu() * mask
                    fake_imgs = generated.detach().cpu() * mask
                    all_real.append(real_imgs)
                    all_fake.append(fake_imgs)

                    acc, f1, iou, fd = self._calc_metrics(batch)
                    epoch_metrics[phase]['precision'].append(acc)
                    epoch_metrics[phase]['fd'].append(fd)
                    epoch_metrics[phase]['f1'].append(f1)
                    epoch_metrics[phase]['iou'].append(iou)

                if all_real and all_fake:
                    real = torch.cat(all_real, dim=0)
                    fake = torch.cat(all_fake, dim=0)
                    fid = self.calculate_fid(real, fake)
                    epoch_metrics[phase]['FID'].append(fid)
                else:
                    epoch_metrics[phase]['FID'].append(0.0)

                self.metrics_history[phase]['precision'].append(np.mean(epoch_metrics[phase]['precision']))
                self.metrics_history[phase]['fd'].append(np.mean(epoch_metrics[phase]['fd']))
                self.metrics_history[phase]['FID'].append(epoch_metrics[phase]['FID'][-1])
                self.metrics_history[phase]['f1'].append(np.mean(epoch_metrics[phase]['f1']))
                self.metrics_history[phase]['iou'].append(np.mean(epoch_metrics[phase]['iou']))

            self._show_epoch_metrics(epoch_metrics)
            self.save_metric_plot(target_metric='FID', suffix='FID')
            self.save_metric_plot(target_metric='fd', suffix='fd')
            self.save_metric_plot(target_metric='f1', suffix='f1')
            self.save_metric_plot(target_metric='iou', suffix='iou')

            if self.model.g_trainer.losses_history['train']:
                # Берем ВСЕ батчи текущей эпохи
                epoch_g_losses = self.model.g_trainer.losses_history['train'][-len(train_loader):]
                
                avg_g_losses = {
                    k: np.mean([batch[k] for batch in epoch_g_losses])
                    for k in ['adversarial_loss', 'bce_loss', 'l1_loss', 'total_loss']
                }
                
                print('Generator Average Losses:')
                print(f'  Adv: {avg_g_losses["adversarial_loss"]} '
                    f'BCE: {avg_g_losses["bce_loss"]} '
                    f'L1: {avg_g_losses["l1_loss"]} '
                    f'Total: {avg_g_losses["total_loss"]}')

            # Для критика
            if self.model.c_trainer.losses_history['train']:
                # Берем ВСЕ батчи текущей эпохи
                epoch_c_losses = self.model.c_trainer.losses_history['train'][-len(train_loader):]
                
                avg_c_losses = {
                    k: np.mean([batch[k] for batch in epoch_c_losses])
                    for k in ['wasserstein_loss', 'gradient_penalty', 'total_loss']
                }
                
                print('Critic Average Losses:')
                print(f'  Wasserstein: {avg_c_losses["wasserstein_loss"]} '
                    f'GP: {avg_c_losses["gradient_penalty"]} '
                    f'Total: {avg_c_losses["total_loss"]}\n')

            if (epoch + 1) % self.checkpoints_ratio == 0:
                self.model.save_checkpoint(self.output_path)

        self.save_test(loader)

    def _process_batch(self, mode: Literal['train', 'valid'], batch: List, visualize_batch=False):
        inputs, targets, masks = [tensor.to(self.device) for tensor in batch]

        if mode == 'train':
            self.model.train_step(inputs, targets, masks)
        else:
            self.model.eval_step(inputs, targets, masks)

        
        with torch.no_grad():
            generated = self.model.generator(inputs, masks)
            if visualize_batch:
                self._visualize_batch(inputs, generated, targets, masks, phase=mode)

        return generated

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
        damaged, target, damage_mask = [el.to(self.device) for el in batch]
        generated = self.model.generator(damaged, damage_mask)

        # Accuracy (по маске)
        gen_bin = (generated.detach().cpu().numpy() > 0.5).astype(np.uint8)
        tgt_bin = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
        mask_bin = (damage_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)

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

        accuracy = np.mean(acc) if acc else 0.0

        # Fractal loss (структурное отличие)
        fd = self._calc_fractal_loss(generated, target, damage_mask)

        return accuracy, np.mean(f1), np.mean(iou), fd

    def _load_checkpoint(self):
        if self.load_weights:
            try:
                _ = self.model.load_checkpoint(self.output_path)
                print(f"Checkpoint загружен успешно. Продолжаем с итерации {self.model.current_iteration}.")
            except FileNotFoundError as e:
                print(f"Ошибка при загрузке чекпоинта: {e}")
                print("Начинаем обучение с нуля.")

    def _calc_fractal_loss(self, generated: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Считает среднюю разницу фрактальной размерности между сгенерированным изображением и ground truth
        """
        fd_total = 0.0
        batch_size = min(generated.shape[0], 4)

        for i in range(batch_size):
            gen_img = generated[i].detach().squeeze()
            tgt_img = target[i].detach().squeeze()

            try:
                fd_gen = FractalAnalyzerGPU.calculate_fractal_dimension(
                    *FractalAnalyzerGPU.box_counting(gen_img)
                )
                fd_target = FractalAnalyzerGPU.calculate_fractal_dimension(
                    *FractalAnalyzerGPU.box_counting(tgt_img)
                )

                fd_total += abs(fd_gen - fd_target)
            except Exception as e:
                print(f"[FD] Ошибка на сэмпле {i}: {e}")
                fd_total += 0.0

        return fd_total / batch_size
    
    def calculate_fid(self, real_images, fake_images):
        """Вычисляет FID между реальными и сгенерированными изображениями."""
        real_images = real_images.to(self.device)
        fake_images = fake_images.to(self.device)
        
        # Масштабирование и интерполяция
        real_image = (real_images > 0.5).float() * 255.0
        fake_image = (fake_images > 0.5).float() * 255.0

        real_image = F.interpolate(real_image, size=(299, 299), mode='bilinear')
        fake_image = F.interpolate(fake_image, size=(299, 299), mode='bilinear')
        
        # Конвертация в 3 канала
        real_image = real_image.repeat(1, 3, 1, 1)
        fake_image = fake_image.repeat(1, 3, 1, 1)
        
        # Нормализация для Inception-v3
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)
        real_image = (real_image/255.0 - mean)/std
        fake_image = (fake_image/255.0 - mean)/std

        # Извлечение признаков
        with torch.no_grad():
            real_features = self.inception_model(real_image)
            fake_features = self.inception_model(fake_image)
        
        # Переводим в numpy
        real_features = real_features.cpu().numpy()
        fake_features = fake_features.cpu().numpy()
        
        # Расчет статистик
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        
        # Вычисление FID
        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real.dot(sigma_fake)).real
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2*covmean)
        
        return fid
    
    def save_test(self, loaders):
        import json
        with open(os.path.join(self.output_path, 'metrics_history.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

        with open(os.path.join(self.output_path, 'losses_history.json'), 'w') as f:
            json.dump({
                'generator': self.model.g_trainer.losses_history,
                'critic': self.model.c_trainer.losses_history
            }, f, indent=4)

        if 'train' in loaders:
            final_batch = next(iter(loaders['train']))
            self._process_batch(mode='train', batch=final_batch, visualize_batch=True)

        if 'valid' in loaders and loaders['valid'] is not None:
            final_val_batch = next(iter(loaders['valid']))
            self._process_batch(mode='valid', batch=final_val_batch, visualize_batch=True)