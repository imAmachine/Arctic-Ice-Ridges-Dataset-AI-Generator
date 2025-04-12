import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import defaultdict


class Metrics:
    """
    Класс для расчета и отслеживания метрик качества восстановления изображений
    """
    
    def __init__(self, name=""):
        """
        Инициализация класса метрик
        
        Args:
            name (str): Название экземпляра для логирования
        """
        self.name = name
        self.current_metrics = {}
        self.history = defaultdict(list)
        self.running_sum = defaultdict(float)
        self.count = 0
    
    def calculate_mse(self, generated, target):
        """
        Расчет Mean Squared Error (MSE)
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            
        Returns:
            float: Значение MSE
        """
        return F.mse_loss(generated, target).item()
    
    def calculate_psnr(self, generated, target, max_val=1.0):
        """
        Расчет Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            max_val (float): Максимальное значение пикселя (по умолчанию 1.0)
            
        Returns:
            float: Значение PSNR
        """
        # Перевод в numpy для использования skimage.metrics
        gen_np = generated.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()
        
        return psnr(target_np, gen_np, data_range=max_val)
    
    def calculate_ssim(self, generated, target):
        """
        Расчет Structural Similarity Index (SSIM)
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            
        Returns:
            float: Значение SSIM
        """
        # Перевод в numpy для использования skimage.metrics
        gen_np = generated.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()
        
        return ssim(target_np, gen_np, data_range=1.0)
    
    def calculate_iou(self, generated, target, threshold=0.5):
        """
        Расчет Intersection over Union (IoU) / коэффициент Жаккара
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            threshold (float): Пороговое значение для бинаризации (по умолчанию 0.5)
            
        Returns:
            float: Значение IoU
        """
        # Бинаризация изображений
        gen_binary = (generated > threshold).float()
        target_binary = (target > threshold).float()
        
        # Рассчитываем пересечение и объединение
        intersection = torch.logical_and(gen_binary, target_binary).sum().float()
        union = torch.logical_or(gen_binary, target_binary).sum().float()
        
        # Избегаем деления на ноль
        if union == 0:
            return 0.0
        
        return (intersection / union).item()
    
    def calculate_dice_coefficient(self, generated, target, threshold=0.5):
        """
        Расчет коэффициента Dice (F1-score)
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            threshold (float): Пороговое значение для бинаризации (по умолчанию 0.5)
            
        Returns:
            float: Значение коэффициента Dice
        """
        # Бинаризация изображений
        gen_binary = (generated > threshold).float()
        target_binary = (target > threshold).float()
        
        # Рассчитываем пересечение
        intersection = torch.logical_and(gen_binary, target_binary).sum().float()
        
        # Сумма элементов в обоих тензорах
        sum_elements = gen_binary.sum() + target_binary.sum()
        
        # Избегаем деления на ноль
        if sum_elements == 0:
            return 0.0
        
        return (2 * intersection / sum_elements).item()
    
    def calculate_mae(self, generated, target):
        """
        Расчет Mean Absolute Error (MAE)
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            
        Returns:
            float: Значение MAE
        """
        return F.l1_loss(generated, target).item()
    
    def calculate_masked_metrics(self, generated, target, mask):
        """
        Расчет метрик только для маскированной области
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение
            target (torch.Tensor): Целевое изображение
            mask (torch.Tensor): Маска (1 - маскированная область, 0 - известная область)
            
        Returns:
            dict: Словарь с метриками для маскированной области
        """
        # Применяем маску к изображениям
        gen_masked = generated * mask
        target_masked = target * mask
        
        # Рассчитываем метрики для маскированной области
        results = {
            'masked_mse': self.calculate_mse(gen_masked, target_masked),
            'masked_mae': self.calculate_mae(gen_masked, target_masked),
            'masked_iou': self.calculate_iou(gen_masked, target_masked),
            'masked_dice': self.calculate_dice_coefficient(gen_masked, target_masked)
        }
        
        return results
    
    def calculate_metrics(self, generated, target, mask=None):
        """
        Расчет всех метрик для одного изображения или батча
        
        Args:
            generated (torch.Tensor): Сгенерированное изображение или батч
            target (torch.Tensor): Целевое изображение или батч
            mask (torch.Tensor, optional): Маска или батч масок
            
        Returns:
            self: Экземпляр класса Metrics с обновленными метриками
        """
        # Если передан батч, обрабатываем его
        if len(generated.shape) == 4:
            return self.calculate_batch_metrics(generated, target, mask)
        
        # Расчет метрик для одного изображения
        self.current_metrics = {
            'mse': self.calculate_mse(generated, target),
            'psnr': self.calculate_psnr(generated, target),
            'ssim': self.calculate_ssim(generated, target),
            'iou': self.calculate_iou(generated, target),
            'dice': self.calculate_dice_coefficient(generated, target),
            'mae': self.calculate_mae(generated, target)
        }
        
        # Если предоставлена маска, добавляем метрики для маскированной области
        if mask is not None:
            masked_metrics = self.calculate_masked_metrics(generated, target, mask)
            self.current_metrics.update(masked_metrics)
        
        # Обновляем историю и накопительные суммы
        self._update_history(self.current_metrics)
        
        return self
    
    def calculate_batch_metrics(self, generated_batch, target_batch, mask_batch=None):
        """
        Расчет всех метрик для батча изображений
        
        Args:
            generated_batch (torch.Tensor): Батч сгенерированных изображений
            target_batch (torch.Tensor): Батч целевых изображений
            mask_batch (torch.Tensor, optional): Батч масок
            
        Returns:
            self: Экземпляр класса Metrics с обновленными метриками
        """
        batch_size = generated_batch.size(0)
        metrics = {
            'mse': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'mae': 0.0
        }
        
        for i in range(batch_size):
            generated = generated_batch[i]
            target = target_batch[i]
            
            metrics['mse'] += self.calculate_mse(generated, target)
            metrics['psnr'] += self.calculate_psnr(generated, target)
            metrics['ssim'] += self.calculate_ssim(generated, target)
            metrics['iou'] += self.calculate_iou(generated, target)
            metrics['dice'] += self.calculate_dice_coefficient(generated, target)
            metrics['mae'] += self.calculate_mae(generated, target)
            
        # Усредняем метрики по батчу
        for key in metrics:
            metrics[key] /= batch_size
            
        # Если предоставлены маски, добавляем метрики для маскированных областей
        if mask_batch is not None:
            masked_metrics = {}
            for i in range(batch_size):
                mask = mask_batch[i]
                result = self.calculate_masked_metrics(
                    generated_batch[i], target_batch[i], mask
                )
                for key, value in result.items():
                    masked_metrics[key] = masked_metrics.get(key, 0.0) + value
            
            for key in masked_metrics:
                masked_metrics[key] /= batch_size
                metrics[key] = masked_metrics[key]
                
        self.current_metrics = metrics
        
        # Обновляем историю и накопительные суммы
        self._update_history(metrics)
        
        return self
    
    def _update_history(self, metrics):
        """
        Обновление истории метрик
        
        Args:
            metrics (dict): Словарь с метриками
        """
        for key, value in metrics.items():
            self.history[key].append(value)
            self.running_sum[key] += value
        self.count += 1
    
    def reset_running_stats(self):
        """
        Сбрасывает накопительную статистику
        """
        self.running_sum = defaultdict(float)
        self.count = 0
    
    def get_average_metrics(self):
        """
        Возвращает усредненные метрики
        
        Returns:
            dict: Словарь с усредненными метриками
        """
        if self.count == 0:
            return {}
        
        avg_metrics = {}
        for key, value in self.running_sum.items():
            avg_metrics[key] = value / self.count
        return avg_metrics
    
    def print_metrics(self, epoch=None, phase=None):
        """
        Вывод текущих метрик
        
        Args:
            epoch (int, optional): Номер эпохи
            phase (str, optional): Фаза (train/val)
        """
        header = self.name
        if epoch is not None:
            header = f"Epoch {epoch}"
        if phase is not None:
            header = f"{header} - {phase}"
        
        metrics_str = self.format_metrics(self.current_metrics)
        print(f"{header}: {metrics_str}")
    
    def print_average_metrics(self, epoch=None, phase=None):
        """
        Вывод усреднённых метрик
        
        Args:
            epoch (int, optional): Номер эпохи
            phase (str, optional): Фаза (train/val)
        """
        avg_metrics = self.get_average_metrics()
        
        header = f"{self.name} Average"
        if epoch is not None:
            header = f"Epoch {epoch}"
        if phase is not None:
            header = f"{header} - {phase}"
        
        metrics_str = self.format_metrics(avg_metrics)
        print(f"{header}: {metrics_str}")
    
    def format_metrics(self, metrics):
        """
        Форматирует метрики для вывода
        
        Args:
            metrics (dict): Словарь с метриками
            
        Returns:
            str: Форматированная строка с метриками
        """
        result = []
        for key, value in metrics.items():
            if 'iou' in key or 'dice' in key or 'ssim' in key:
                result.append(f"{key}: {value:.4f}")
            else:
                result.append(f"{key}: {value:.6f}")
        
        return " | ".join(result)
    
    def get_history(self, metric_name):
        """
        Возвращает историю конкретной метрики
        
        Args:
            metric_name (str): Название метрики
            
        Returns:
            list: История значений метрики
        """
        return self.history.get(metric_name, [])
    
    def plot_history(self, metric_names=None, figsize=(10, 6), title=None):
        """
        Построение графика истории метрик
        
        Args:
            metric_names (list, optional): Список названий метрик для отображения
            figsize (tuple, optional): Размер графика
            title (str, optional): Заголовок графика
        """
        import matplotlib.pyplot as plt
        
        if metric_names is None:
            metric_names = ['ssim', 'iou', 'dice', 'psnr']
            metric_names = [name for name in metric_names if name in self.history]
        
        plt.figure(figsize=figsize)
        
        for metric_name in metric_names:
            if metric_name in self.history:
                plt.plot(self.history[metric_name], label=metric_name)
        
        if title is None:
            title = f"{self.name} Metrics History"
        
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()