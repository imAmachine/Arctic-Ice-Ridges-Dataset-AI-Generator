import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

from src.gan.metrics import Metrics
from src.common.analyze_tools import FractalAnalyzerGPU
from src.common.interfaces import IModelTrainer
from src.gan.arch import WGanDiscriminator, WGanGenerator

class GenerativeModel:
    def __init__(self, target_image_size=448, g_feature_maps=64, d_feature_maps=16, device='cpu'):
        self.device = device
        self.generator = WGanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.discriminator = WGanDiscriminator(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        self.g_metrics = Metrics(name="Generator")
        self.d_metrics = Metrics(name="Discriminator")
        
        # Параметры WGAN
        self.n_train_critic = 5
        self.critic_clip_value = 0.01
        self.current_iteration = 0
        
        self.g_trainer, self.c_trainer = self._init_trainers()
        self.target_image_size = target_image_size

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def _init_trainers(self):
        g_trainer = WGANGeneratorModelTrainer(model=self.generator, critic=self.discriminator)
        c_trainer = WGANCriticModelTrainer(model=self.discriminator, clip_value=self.critic_clip_value)
        return g_trainer, c_trainer
    
    def train_step(self, inputs, targets, masks):
        self.current_iteration += 1
        
        # Обучение критика несколько раз
        c_loss_dict = {}
        for _ in range(self.n_train_critic):
            with torch.no_grad():
                fake_images = self.generator(inputs, masks)
            c_loss_dict = self.c_trainer.step(targets, fake_images, masks)
        
        # Одно обучение генератора
        g_loss_dict, fake_images = self.g_trainer.step(inputs, targets, masks)
        
        return {'g_losses': g_loss_dict, 'd_losses': c_loss_dict}
    
    def _save_models(self, output_path):
        self.g_trainer.save_model_state_dict(output_path)
        self.c_trainer.save_model_state_dict(output_path)

    def _load_weights(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        
        gen_path = os.path.join(output_path, 'generator.pt')
        critic_path = os.path.join(output_path, 'critic.pt')
        if os.path.exists(gen_path) and os.path.exists(critic_path):
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True))
            self.discriminator.load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=True))
        else:
            raise FileNotFoundError('Ошибка загрузки весов моделей')


class WGANGeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, critic):
        self.model = model
        self.critic = critic

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.l1_criterion = nn.SmoothL1Loss()
        self.loss_history = []
        self.loss_history_val = []

    def _calc_adv_loss(self, generated, mask):
        fake_pred_damaged = self.critic(generated * mask)
        fake_pred_known = self.critic(generated * (1 - mask))
        
        adv_loss_damaged = -torch.mean(fake_pred_damaged)
        adv_loss_known = -torch.mean(fake_pred_known)
        
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
        
        # Wasserstein loss для генератора
        w_loss = self._calc_adv_loss(generated, mask)
        
        # L1 loss для пиксельной разницы в областях с маской (для улучшения качества)
        l1_loss = self.l1_criterion(target * mask, generated * mask)
        
        # Полная потеря с весовыми коэффициентами
        total_loss = w_loss + 10.0 * l1_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'w_loss': w_loss.item(),
            'l1_loss': l1_loss.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict, generated


class WGANCriticModelTrainer(IModelTrainer):
    def __init__(self, model, clip_value=0.01):
        self.model = model
        self.clip_value = clip_value

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.loss_history = []
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "critic.pt"))
    
    def _calc_wasserstein_loss(self, real_target, fake_generated, masks):
        # Критик на реальных данных
        real_pred_known = self.model(real_target * (1 - masks))
        real_pred_damaged = self.model(real_target * masks)
        
        # Критик на сгенерированных данных
        fake_pred_known = self.model(fake_generated * (1 - masks))
        fake_pred_damaged = self.model(fake_generated * masks)
        
        real_loss = -torch.mean(real_pred_known) - torch.mean(real_pred_damaged)
        fake_loss = torch.mean(fake_pred_known) + torch.mean(fake_pred_damaged)
        
        wasserstein_loss = real_loss + fake_loss
        
        return wasserstein_loss
    
    def step(self, real_target, fake_generated, masks):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Расчет потери Wasserstein
        wasserstein_loss = self._calc_wasserstein_loss(
            real_target=real_target, 
            fake_generated=fake_generated.detach(),
            masks=masks
        )
        
        wasserstein_loss.backward()
        self.optimizer.step()
        
        # Clip weights для критика после оптимизации
        for p in self.model.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
        
        loss_dict = {
            'total_loss': wasserstein_loss.item()
        }
        self.loss_history.append(loss_dict)
        return loss_dict