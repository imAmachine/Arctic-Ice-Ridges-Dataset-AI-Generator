from collections import defaultdict
import os
from typing import Dict, Literal
import torch
import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

from src.gan.dataset import IceRidgeDataset
from src.common.interfaces import IGenerativeModel, IModelTrainer
from src.gan.arch import WGanCritic, WGanGenerator

class GenerativeModel(IGenerativeModel):
    def __init__(self, device: str, 
                 losses_weights: Dict,
                 optimization_params: Dict,
                 target_image_size,
                 g_feature_maps, 
                 d_feature_maps, 
                 n_critic):
        super().__init__(target_image_size, device, optimization_params)
        self.generator = WGanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.critic = WGanCritic(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        self.current_iteration = 0

        self.n_critic = n_critic
        self.g_trainer, self.c_trainer = self._init_trainers(losses_weights.get('gen'), losses_weights.get('discr'))

    def _init_trainers(self, g_losses_weights, c_losses_weights) -> tuple['WGANGeneratorModelTrainer', 'WGANCriticModelTrainer']:
        g_trainer = WGANGeneratorModelTrainer(
            model=self.generator, 
            critic=self.critic,
            losses_weights=g_losses_weights
        )
        
        c_trainer = WGANCriticModelTrainer(
            model=self.critic, 
            losses_weights=c_losses_weights
        )
        
        for trainer in [g_trainer, c_trainer]:
            trainer.optimizer = torch.optim.RMSprop(trainer.model.parameters(), lr=self.optimization_params.get('lr'))
            trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode=self.optimization_params.get('mode'), factor=0.5, patience=6)
        
        return g_trainer, c_trainer

    def get_model_transforms(self) -> 'transforms.Compose':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def switch_mode(self, mode: Literal['train', 'valid']='train') -> None:
        self.generator.train() if mode == 'train' else self.generator.eval()
        self.critic.train() if mode == 'train' else self.critic.eval()
    
    def _train_critic_step(self, input_data, target_data, damage_mask):
        with torch.no_grad():
            for _ in range(self.n_critic):
                fake_images = self.generator(input_data, damage_mask)
        self.c_trainer.train_step(target_data, fake_images.detach())
    
    def _train_generator_step(self, input_data, target_data, damage_mask):
        generated = self.generator(input_data, damage_mask)
        self.g_trainer.train_step(generated, target_data)
    
    def train_step(self, input_data, target_data, damage_mask) -> None:
        self.current_iteration += 1
        
        # цикл обучения критика
        self._train_critic_step(input_data, target_data, damage_mask)
        
        # шаг обучения генератора
        self._train_generator_step(input_data, target_data, damage_mask)

    def eval_step(self, input, target, damage_mask) -> None:
        generated = self.g_trainer.eval_step(input, target, damage_mask)
        self.c_trainer.eval_step(target, generated.detach())

    def step_schedulers(self, metric: float) -> None:
        self.g_trainer.step_scheduler(metric)
        self.c_trainer.step_scheduler(metric)

    def save_checkpoint(self, output_path): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
        os.makedirs(output_path, exist_ok=True)
        checkpoint = {
            'current_iteration': self.current_iteration,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optimizer': self.g_trainer.optimizer.state_dict(),
            'critic_optimizer': self.c_trainer.optimizer.state_dict(),
            'generator_scheduler': self.g_trainer.scheduler.state_dict(),
            'critic_scheduler': self.c_trainer.scheduler.state_dict(),
            'generator_loss_history': self.g_trainer.loss_history,
            'critic_loss_history': self.c_trainer.loss_history
        }
        torch.save(checkpoint, os.path.join(output_path, 'training_checkpoint.pt'))
        print(f"Checkpoint сохранен в {os.path.join(output_path, 'training_checkpoint.pt')}")

    def load_checkpoint(self, path): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
        checkpoint_path = os.path.join(path, 'training_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])

            self.g_trainer.optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.c_trainer.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.g_trainer.scheduler.load_state_dict(checkpoint['generator_scheduler'])
            self.c_trainer.scheduler.load_state_dict(checkpoint['critic_scheduler'])

            self.g_trainer.loss_history = checkpoint['generator_loss_history']
            self.c_trainer.loss_history = checkpoint['critic_loss_history']
            self.current_iteration = checkpoint['current_iteration']

            print(f"Checkpoint загружен из {checkpoint_path} (итерация {self.current_iteration})")
            return True
        else:
            raise FileNotFoundError(f"Checkpoint файл не найден по пути {checkpoint_path}")

    def infer_generate(self, preprocessed_img, checkpoint_path, processor): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
        self.switch_mode('eval')
        self.load_checkpoint(checkpoint_path)

        damaged, original, outpaint_mask = IceRidgeDataset.prepare_data(
            img=preprocessed_img,
            processor=processor,
            augmentations=None,
            model_transforms=self.get_model_transforms()
        )
        damaged = damaged.to(self.device)
        outpaint_mask = outpaint_mask.to(self.device)

        with torch.no_grad():
            generated = self.generator(damaged.unsqueeze(1), outpaint_mask.unsqueeze(1))

        generated_img = generated.detach().cpu().squeeze().numpy() * 255
        original_img = original.squeeze().numpy() * 255

        return generated_img, original_img


class WGANGeneratorModelTrainer(IModelTrainer):
    def __init__(self, model, critic, losses_weights: Dict):
        super().__init__(model, losses_weights)
        self.critic = critic
        self.criterion = self._calc_losses
    
    def _calc_adv_loss(self, input_data):
        return -torch.mean(self.critic(input_data))
    
    def _calc_losses(self, input_data, target_data, phase='train') -> None:
        self.loss_history[phase].append({})
        self.calc_loss(loss_fn=self._calc_adv_loss, loss_name='adv', phase=phase, samples=(input_data,))
        self.calc_loss(loss_fn=nn.BCELoss(), loss_name='bce', phase=phase, samples=(input_data, target_data))
            
    def train_step(self, generated, target) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        
        self.total_train_loss = torch.tensor(0.0, device="cuda:0")
        self.criterion(generated, target, phase='train')
        self.total_train_loss.backward()
        
        self.optimizer.step()
    
    def eval_step(self, input, target, mask):
        self.model.eval()
        with torch.no_grad():
            generated = self.model(input, mask)
            self.criterion(generated, target, phase='valid')
            return generated


class WGANCriticModelTrainer(IModelTrainer):
    def __init__(self, model, losses_weights: Dict):
        super().__init__(model, losses_weights)
    
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
    
    def _calc_wass_loss(self, real_pred, fake_pred):
        return -torch.mean(real_pred) + torch.mean(fake_pred)
    
    def _calc_losses(self, real_target, fake_generated, phase='train'):
        self.loss_history[phase].append({})
        
        real_pred = self.model(real_target)
        fake_pred = self.model(fake_generated)
        
        self.calc_loss(loss_fn=self._calc_wass_loss, loss_name='wasserstein', phase=phase, samples=(real_pred, fake_pred))
        
        if phase == 'train':
            self.calc_loss(loss_fn=self._calc_gradient_penalty, loss_name='gp', phase=phase, samples=(real_target, fake_generated))    

    def train_step(self, real_target, fake_generated) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        
        self.total_train_loss = torch.tensor(0.0, device="cuda:0")
        self.criterion(real_target, fake_generated, phase='train')
        self.total_train_loss.backward()
        
        self.optimizer.step()

    def eval_step(self, real_target, fake_generated):
        self.model.eval()
        with torch.no_grad():
            self.criterion(real_target, fake_generated, phase='valid')
