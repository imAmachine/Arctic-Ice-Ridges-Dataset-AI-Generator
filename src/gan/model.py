import os
from typing import Dict, Literal, Tuple, Type
import torch
import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

from src.common.interfaces import IModelTrainer
from src.gan.arch import WGanCritic, WGanGenerator

class GenerativeModel:
    def __init__(self, target_image_size=256, 
                 g_feature_maps=64, 
                 d_feature_maps=32, 
                 device='cpu',
                 n_critic=3,
                 lambda_gp=10,
                 lr=0.0002,
                 lambda_w=0.5,
                 lambda_bce=1.2,
                 lambda_l1=1.0):
        self.device = device
        self.generator = WGanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.critic = WGanCritic(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        self.target_image_size = target_image_size
        
        # Параметры для WGAN
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.current_iteration = 0
        
        self.learning_rate = lr
        self.lambda_w = lambda_w
        self.lambda_bce = lambda_bce
        self.lambda_l1 = lambda_l1
        
        self.g_trainer, self.c_trainer = self._init_trainers()

    def model_transforms(self) -> Type['transforms.Compose']:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size), 
                              interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
    
    def switch_mode(self, mode: Literal['train', 'eval'] = 'train') -> None:
        if mode == 'train':
            self.generator.train()
            self.critic.train()
        
        if mode == 'eval':
            self.generator.eval()
            self.critic.eval()
    
    def _init_trainers(self) -> Tuple[Type['WGANGeneratorModelTrainer'], Type['WGANCriticModelTrainer']]:
        g_trainer = WGANGeneratorModelTrainer(model=self.generator, 
                                              critic=self.critic, 
                                              lr=self.learning_rate,
                                              lambda_w=self.lambda_w,
                                              lambda_l1=self.lambda_l1,
                                              lambda_bce=self.lambda_bce)
        
        c_trainer = WGANCriticModelTrainer(model=self.critic, 
                                           lambda_gp=self.lambda_gp,
                                           lr=self.learning_rate)
        
        return g_trainer, c_trainer
    
    def train_step(self, damaged, original, damage_mask) -> Dict[Literal['g_losses', 'd_losses'], Dict]:
        self.current_iteration += 1
        
        c_loss_dict = {}
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake_images = self.generator(damaged, damage_mask) # генерация изображения

            c_loss_dict = self.c_trainer.step(original, fake_images) # обучение критика
        g_loss_dict, fake_images = self.g_trainer.step(damaged, original, damage_mask) # обучение генератора
        return {'g_losses': g_loss_dict, 'd_losses': c_loss_dict}
    
    def save_checkpoint(self, output_path):
        """Сохраняет полное состояние обучения в checkpoint файл"""
        checkpoint = {
            'current_iteration': self.current_iteration,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optimizer': self.g_trainer.optimizer.state_dict(),
            'critic_optimizer': self.c_trainer.optimizer.state_dict(),
            'generator_scheduler': self.g_trainer.scheduler.state_dict(),
            'critic_scheduler': self.c_trainer.scheduler.state_dict(),
            'generator_loss_history': self.g_trainer.loss_history,
            'critic_loss_history': self.c_trainer.loss_history,
            'generator_loss_history_val': self.g_trainer.loss_history_val
        }
        
        os.makedirs(output_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(output_path, 'training_checkpoint.pt'))
        print(f"Checkpoint сохранен в {os.path.join(output_path, 'training_checkpoint.pt')}")

    def load_checkpoint(self, output_path):
        """Загружает полное состояние обучения из checkpoint файла"""
        checkpoint_path = os.path.join(output_path, 'training_checkpoint.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Загрузка весов моделей
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            # Загрузка состояния оптимизаторов
            self.g_trainer.optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.c_trainer.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
            # Загрузка состояния планировщиков
            self.g_trainer.scheduler.load_state_dict(checkpoint['generator_scheduler'])
            self.c_trainer.scheduler.load_state_dict(checkpoint['critic_scheduler'])
            
            # Загрузка истории потерь
            self.g_trainer.loss_history = checkpoint['generator_loss_history']
            self.c_trainer.loss_history = checkpoint['critic_loss_history']
            self.g_trainer.loss_history_val = checkpoint['generator_loss_history_val']
            
            # Загрузка счетчика итераций
            self.current_iteration = checkpoint['current_iteration']
            
            print(f"Checkpoint загружен из {checkpoint_path} (итерация {self.current_iteration})")
            return True
        else:
            raise FileNotFoundError(f"Checkpoint файл не найден по пути {checkpoint_path}")

    def _load_weights(self, output_path):
        """Загружает только веса моделей (устаревший метод)"""
        os.makedirs(output_path, exist_ok=True)
        
        # Сначала проверим наличие checkpoint файла
        checkpoint_path = os.path.join(output_path, 'training_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            try:
                return self.load_checkpoint(output_path)
            except Exception as e:
                print(f"Ошибка загрузки checkpoint: {e}. Пробуем загрузить только веса моделей.")
        
        # Если checkpoint не найден или не удалось загрузить, пробуем загрузить отдельные файлы с весами
        gen_path = os.path.join(output_path, 'generator.pt')
        critic_path = os.path.join(output_path, 'critic.pt')
        if os.path.exists(gen_path) and os.path.exists(critic_path):
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device, weights_only=True))
            print("Загружены только веса моделей (без состояния обучения)")
            return False
        else:
            raise FileNotFoundError('Ошибка загрузки весов моделей')

    def _save_models(self, output_path):
        """Сохраняет только веса моделей (устаревший метод)"""
        self.g_trainer.save_model_state_dict(output_path)
        self.c_trainer.save_model_state_dict(output_path)
        self.save_checkpoint(output_path)


class WGANGeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, critic, lambda_w, lambda_bce, lambda_l1, lr):
        self.model = model
        self.critic = critic

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=6, verbose=True
        )

        self._BCE = nn.BCELoss()
        self._L1 = nn.L1Loss()
        
        self.lambda_w = lambda_w
        self.lambda_bce = lambda_bce
        self.lambda_l1 = lambda_l1
        
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
        bce_masked = self._BCE(generated, target)
        l1_context = self._L1(generated, target)
        total_loss = w_loss * self.lambda_w + bce_masked * self.lambda_bce + l1_context * self.lambda_l1
        
        total_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'w_loss': w_loss.item(),
            'bce_masked': bce_masked.item(),
            'l1_context': l1_context.item()
        }
        self.loss_history.append(loss_dict)
        
        return loss_dict, generated


class WGANCriticModelTrainer(IModelTrainer):
    def __init__(self, model, lambda_gp, lr):
        self.model = model
        self.lambda_gp = lambda_gp
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=6, verbose=True
        )
        self.loss_history = []
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "critic.pt"))
    
    def _calc_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)

        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates.requires_grad_(True)
        d_interpolates = self.model(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()
    
    def _calc_wasserstein_loss(self, real_target, fake_generated):
        real_pred = self.model(real_target)
        fake_pred = self.model(fake_generated)
        wasserstein_loss = -torch.mean(real_pred) + torch.mean(fake_pred)
        return wasserstein_loss
    
    def step(self, real_target, fake_generated):
        self.model.train()
        self.optimizer.zero_grad()
        
        wasserstein_loss = self._calc_wasserstein_loss(real_target=real_target, fake_generated=fake_generated.detach())
        grad_penalty = self._calc_gradient_penalty(real_samples=real_target, fake_samples=fake_generated.detach())
        
        total_loss = wasserstein_loss + grad_penalty * self.lambda_gp
        
        total_loss.backward()
        self.optimizer.step()
                
        loss_dict = {
            'total_loss': total_loss.item(),
            'wasserstein_loss': wasserstein_loss.item(),
            'gradient_penalty': grad_penalty.item()
        }
        self.loss_history.append(loss_dict)
        return loss_dict