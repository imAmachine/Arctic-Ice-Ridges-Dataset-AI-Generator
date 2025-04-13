import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

from src.common.interfaces import IModelTrainer
from src.gan.arch import WGanCritic, WGanGenerator

class GenerativeModel:
    def __init__(self, target_image_size=448, g_feature_maps=64, d_feature_maps=64, device='cpu'):
        self.device = device
        self.generator = WGanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.critic = WGanCritic(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        self.target_image_size = target_image_size
        
        # Параметры для WGAN
        self.n_critic = 5
        self.clip_value = 0.01
        self.current_iteration = 0
        
        self.g_trainer, self.c_trainer = self._init_trainers()

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])
    
    def _init_trainers(self):
        g_trainer = WGANGeneratorModelTrainer(model=self.generator, critic=self.critic)
        c_trainer = WGANCriticModelTrainer(model=self.critic, clip_value=self.clip_value)
        
        return g_trainer, c_trainer
    
    def train_step(self, inputs, targets, masks):
        self.current_iteration += 1
        
        c_loss_dict = {}
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake_images = self.generator(inputs, masks)
            c_loss_dict = self.c_trainer.step(targets, fake_images)
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
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=True))
        else:
            raise FileNotFoundError('Ошибка загрузки весов моделей')


class WGANGeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, critic):
        self.model = model
        self.critic = critic

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=6, verbose=True
        )

        self.bce_criterion = nn.BCELoss()
        self.loss_history = []
        self.loss_history_val = []

    def _calc_adv_loss(self, generated):
        fake_pred_damaged = self.critic(generated)
        return -torch.mean(fake_pred_damaged)

    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))

    def step(self, damaged, target, mask):
        self.model.train()
        self.optimizer.zero_grad()
        generated = self.model(damaged, mask)
        
        w_loss = self._calc_adv_loss(generated)
        
        bce_loss = self.bce_criterion(generated * mask, target * mask)
        total_loss = w_loss + bce_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'w_loss': w_loss.item(),
            'bce_loss': bce_loss.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict, generated


class WGANCriticModelTrainer(IModelTrainer):
    def __init__(self, model, clip_value=0.01):
        self.model = model
        self.clip_value = clip_value

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=6, verbose=True
        )
        self.loss_history = []
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "critic.pt"))
    
    def _calc_wasserstein_loss(self, real_target, fake_generated):
        real_pred = self.model(real_target)
        fake_pred = self.model(fake_generated.detach())
        
        wasserstein_loss = -torch.mean(real_pred) + torch.mean(fake_pred)
        
        return wasserstein_loss
    
    def step(self, real_target, fake_generated):
        self.model.train()
        self.optimizer.zero_grad()
        
        wasserstein_loss = self._calc_wasserstein_loss(
            real_target=real_target, 
            fake_generated=fake_generated
        )
        
        wasserstein_loss.backward()
        self.optimizer.step()
        
        for p in self.model.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
        
        loss_dict = {
            'total_loss': wasserstein_loss.item()
        }
        self.loss_history.append(loss_dict)
        return loss_dict