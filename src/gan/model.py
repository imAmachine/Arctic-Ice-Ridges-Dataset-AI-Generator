import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms, InterpolationMode

from src.common.analyze_tools import FractalAnalyzerGPU
from src.common.interfaces import IModelTrainer
from gan.arch import GanDiscriminator, GanGenerator

class GenerativeModel:
    def __init__(self, target_image_size=448, g_feature_maps=64, d_feature_maps=16, device='cpu'):
        self.device = device
        self.generator = GanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.discriminator = GanDiscriminator(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        
        self.g_trainer, self.d_trainer = self._init_trainers()
        
        self.target_image_size = target_image_size

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def _init_trainers(self):
        g_trainer = GeneratorModelTrainer(model=self.generator,discriminator=self.discriminator)
        d_trainer = DiscriminatorModelTrainer(model=self.discriminator)
        
        return g_trainer, d_trainer
    
    def train_step(self, inputs, targets, masks):
        g_loss_dict, fake_images = self.g_trainer.step(inputs, targets, masks)        
        d_loss_dict = self.d_trainer.step(targets, fake_images, masks)
        
        return {'g_losses': g_loss_dict, 'd_losses': d_loss_dict}
    
    def _save_models(self, output_path):
        self.g_trainer.save_model_state_dict(output_path)
        self.d_trainer.save_model_state_dict(output_path)

    def _load_weights(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        
        gen_path = os.path.join(output_path, 'generator.pt')
        discr_path = os.path.join(output_path, 'discriminator.pt')
        if os.path.exists(gen_path) and os.path.exists(discr_path):
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True))
            self.discriminator.load_state_dict(torch.load(discr_path, map_location=self.device, weights_only=True))
        else:
            raise FileNotFoundError('Ошибка загрузки весов моделей')


class GeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, discriminator):
        self.model = model
        self.discriminator = discriminator
        # Изменяем параметры оптимизатора - меньшая скорость обучения
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Критерии потерь
        self.adv_criterion = nn.BCELoss()
        self.bce_criterion = nn.BCELoss()
        self.l1_criterion = nn.L1Loss()
        self.loss_history = []
        self.loss_history_val = []

        # Сбалансированные веса для различных компонентов функции потерь
        self.adv_loss_weight = 1.0
        self.l1_loss_weight = 1.0
        self.bce_loss_weight = 1.0
        self.mask_loss_weight = 1.0
        self.fractal_loss_weight = 0.5

    def _calc_adv_loss(self, generated, mask):
        # Добавляем небольшой шум к меткам для стабилизации обучения
        fake_pred_damaged = self.discriminator(generated * mask)
        real_label_damaged = torch.ones_like(fake_pred_damaged).uniform_(0.8, 1.0)

        fake_pred_known = self.discriminator(generated * (1 - mask))
        real_label_known = torch.ones_like(fake_pred_known).uniform_(0.8, 1.0)
        
        adv_loss_damaged = self.adv_criterion(fake_pred_damaged, real_label_damaged)
        adv_loss_known = self.adv_criterion(fake_pred_known, real_label_known)
        
        total_adv_loss = adv_loss_known + adv_loss_damaged
        return total_adv_loss

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))
    
    def _calc_fractal_loss(self, generated, mask):
        fd_losses = 0.0
        batch_size = min(generated.shape[0], 2)
        
        for i in range(batch_size):
            img = generated[i].squeeze()
            m = mask[i].squeeze()
            
            # Уменьшаем размер для ускорения вычислений
            if img.shape[0] > 128:
                img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(128, 128), 
                                   mode='bilinear', align_corners=False).squeeze()
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(128, 128), 
                                 mode='nearest').squeeze()
            
            part_masked = img * m
            part_unmasked = img * (1 - m)
            
            try:
                fd_masked = FractalAnalyzerGPU.calculate_fractal_dimension(
                    *FractalAnalyzerGPU.box_counting(part_masked)
                )
                fd_unmasked = FractalAnalyzerGPU.calculate_fractal_dimension(
                    *FractalAnalyzerGPU.box_counting(part_unmasked)
                )
                
                fd_losses += abs(fd_masked - fd_unmasked)
            except Exception as e:
                print(f"Ошибка расчета фракталов: {e}")
                fd_losses += 0.0
                
        fd_loss = fd_losses / batch_size
        return fd_loss

    def step(self, input_masked, target, mask):
        self.model.train()
        self.optimizer.zero_grad()
        generated = self.model(input_masked, mask)
        
        # Вычисляем BCE loss между сгенерированными и целевыми изображениями
        bce_loss = self.bce_criterion(generated, target)
        
        # Потеря для известных областей
        mask_loss = self.adv_criterion(generated * mask, target * mask)
        
        # Adversarial loss
        adv_loss = self._calc_adv_loss(generated, mask)
        
        # L1 loss для пиксельной разницы в областях с маской
        l1_loss = self.l1_criterion(target * mask, generated * mask)
        
        # Активируем фрактальную потерю
        fd_loss = self._calc_fractal_loss(generated, mask)
        
        # Суммарный total_loss с учётом весовых коэффициентов
        total_loss = (self.adv_loss_weight * adv_loss +
                      self.l1_loss_weight * l1_loss +
                      self.bce_loss_weight * bce_loss +
                      self.mask_loss_weight * mask_loss +
                      self.fractal_loss_weight * fd_loss)
        
        total_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'adv_loss': adv_loss.item(),
            'l1_loss': l1_loss.item(),
            'bce_loss': bce_loss.item(),
            'mask_loss': mask_loss.item(),
            'fractal_loss': fd_loss
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict, generated


class DiscriminatorModelTrainer(IModelTrainer):
    def __init__(self, model, optimizer=None):
        self.model = model
        # Уменьшаем скорость обучения
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.BCELoss()
        self.loss_history = []
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "discriminator.pt"))
    
    def _calc_adv_loss(self, real_target, fake_generated, masks):
        real_pred_known = self.model(real_target * (1 - masks))
        real_label_known = torch.ones_like(real_pred_known).uniform_(0.8, 1.0)

        real_pred_damaged = self.model(real_target * masks)
        real_label_damaged = torch.ones_like(real_pred_damaged).uniform_(0.8, 1.0)

        fake_pred_known = self.model(fake_generated * (1 - masks))
        fake_label_known = torch.zeros_like(fake_pred_known).uniform_(0.0, 0.2)

        fake_pred_damaged = self.model(fake_generated * masks)
        fake_label_damaged = torch.zeros_like(fake_pred_damaged).uniform_(0.0, 0.2)

        weight = 0.8
        
        real_loss = weight * (self.criterion(real_pred_known, real_label_known) + 
                     self.criterion(real_pred_damaged, real_label_damaged))
        fake_loss = weight * (self.criterion(fake_pred_known, fake_label_known) + 
                     self.criterion(fake_pred_damaged, fake_label_damaged))

        total_adv_loss = real_loss + fake_loss
        return total_adv_loss
    
    def step(self, real_target, fake_generated, masks):
        self.model.train()
        self.optimizer.zero_grad()
        total_adv_loss = self._calc_adv_loss(real_target=real_target, 
                                             fake_generated=fake_generated.detach(),
                                             masks=masks)

        total_adv_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'total_loss': total_adv_loss.item()
        }
        self.loss_history.append(loss_dict)
        return loss_dict