import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

from src.gan.dataset import IceRidgeDataset
from src.common.interfaces import IModelTrainer
from src.gan.arch import WGanCritic, WGanGenerator

class GenerativeModel_GAN:
    def __init__(self, target_image_size, g_feature_maps, d_feature_maps, device,
                 n_critic, lambda_gp, lr, lambda_w, lambda_bce, lambda_l1):
        self.device = device
        self.generator = WGanGenerator(input_channels=2, feature_maps=g_feature_maps).to(self.device)
        self.critic = WGanCritic(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        self.target_image_size = target_image_size
        self.current_iteration = 0

        self.n_critic = n_critic
        self.learning_rate = lr

        self.lambda_gp = lambda_gp
        self.lambda_w = lambda_w
        self.lambda_bce = lambda_bce
        self.lambda_l1 = lambda_l1

        self.g_trainer, self.c_trainer = self._init_trainers()

    def _init_trainers(self):
        g_trainer = WGANGeneratorModelTrainer(
            model=self.generator, 
            critic=self.critic, 
            lr=self.learning_rate,
            lambda_w=self.lambda_w,
            lambda_l1=self.lambda_l1,
            lambda_bce=self.lambda_bce,
        )
        c_trainer = WGANCriticModelTrainer(
            model=self.critic, 
            lambda_gp=self.lambda_gp,
            lr=self.learning_rate
        )
        return g_trainer, c_trainer

    def get_model_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_image_size, self.target_image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def switch_mode(self, mode='train'):
        self.generator.train() if mode == 'train' else self.generator.eval()
        self.critic.train() if mode == 'train' else self.critic.eval()

    def train_step(self, input, target, damage_mask) -> None:
        self.current_iteration += 1
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake_images = self.generator(input, damage_mask)
            self.c_trainer.train_step(target, fake_images.detach())
        self.g_trainer.train_step(input, target, damage_mask)

    def eval_step(self, input, target, damage_mask) -> None:
        with torch.no_grad():
            generated = self.g_trainer.eval_step(input, target, damage_mask)
            self.c_trainer.eval_step(target, generated.detach())

    def step_schedulers(self, val_loss):
        self.g_trainer.step_scheduler(val_loss)
        self.c_trainer.step_scheduler(val_loss)

    def reset_all_losses(self):
        self.g_trainer.reset_losses()
        self.c_trainer.reset_losses()

    def save_checkpoint(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        checkpoint = {
            'current_iteration': self.current_iteration,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optimizer': self.g_trainer.optimizer.state_dict(),
            'critic_optimizer': self.c_trainer.optimizer.state_dict(),
            'generator_scheduler': self.g_trainer.scheduler.state_dict(),
            'critic_scheduler': self.c_trainer.scheduler.state_dict(),
            'generator_loss_history': self.g_trainer.losses_history,
            'critic_loss_history': self.c_trainer.losses_history
        }
        torch.save(checkpoint, os.path.join(output_path, 'training_checkpoint.pt'))
        print(f"Checkpoint сохранен в {os.path.join(output_path, 'training_checkpoint.pt')}")

    def load_checkpoint(self, output_path):
        checkpoint_path = os.path.join(output_path, 'training_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])

            self.g_trainer.optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.c_trainer.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.g_trainer.scheduler.load_state_dict(checkpoint['generator_scheduler'])
            self.c_trainer.scheduler.load_state_dict(checkpoint['critic_scheduler'])

            self.g_trainer.losses_history = checkpoint['generator_loss_history']
            self.c_trainer.losses_history = checkpoint['critic_loss_history']
            self.current_iteration = checkpoint['current_iteration']

            print(f"Checkpoint загружен из {checkpoint_path} (итерация {self.current_iteration})")
            return True
        else:
            raise FileNotFoundError(f"Checkpoint файл не найден по пути {checkpoint_path}")

    def infer_generate(self, preprocessed_img, checkpoint_path, processor):
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
    def __init__(self, model, critic, lambda_w, lambda_bce, lambda_l1, lr):
        self.model = model
        self.critic = critic

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=6)
        
        self.criterion = self._calc_losses
        self.losses_history = {'train': [], 'valid': []}
        
        # losses weights
        self.lambda_w = lambda_w
        self.lambda_bce = lambda_bce
        self.lambda_l1 = lambda_l1
        
    def _calc_losses(self, input, target, phase='train'):
        adversarial_loss = -torch.mean(self.critic(input)) * self.lambda_w
        bce_loss = nn.BCELoss()(input, target) * self.lambda_bce
        l1_loss = nn.L1Loss()(input, target) * self.lambda_l1
        total_loss = adversarial_loss + bce_loss + l1_loss
        
        self.losses_history[phase].append({
            'adversarial_loss': adversarial_loss.item(),
            'bce_loss': bce_loss.item(), 
            'l1_loss': l1_loss.item(),
            'total_loss': total_loss.item()
        })
        
        return total_loss
            
    def train_step(self, input, target, mask):
        self.model.train()
        self.optimizer.zero_grad()
        
        generated = self.model(input, mask)
        loss = self.criterion(generated, target, phase='train')
        loss.backward()
        self.optimizer.step()
        
        return generated
    
    def eval_step(self, input, target, mask):
        self.model.eval()
        with torch.no_grad():
            generated = self.model(input, mask)
            _ = self.criterion(generated, target, phase='valid')
            return generated
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "generator.pt"))
    
    def load_model_state_dict(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"Загружено состояние генератора из: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Файл {checkpoint_path} не найден.")

    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)
    
    def reset_losses(self):
        self.losses_history = {'train': [], 'valid': []}


class WGANCriticModelTrainer(IModelTrainer):
    def __init__(self, model, lambda_gp, lr):
        self.model = model
        self.lambda_gp = lambda_gp

        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=6)

        self.criterion = self._calc_losses
        self.losses_history = {'train': [], 'valid': []}
    
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
    
    def _calc_losses(self, real_target, fake_generated, phase='train'):
        real_pred = self.model(real_target)
        fake_pred = self.model(fake_generated)

        wasserstein_loss = -torch.mean(real_pred) + torch.mean(fake_pred)

        grad_penalty = self._calc_gradient_penalty(real_target, fake_generated) if phase == 'train' else torch.tensor(0.0, device=real_target.device)
        total_loss = wasserstein_loss + grad_penalty * self.lambda_gp

        self.losses_history[phase].append({
            'total_loss': total_loss.item(),
            'wasserstein_loss': wasserstein_loss.item(),
            'gradient_penalty': grad_penalty.item() if phase == 'train' else 0.0
        })

        return total_loss
    
    def train_step(self, real_target, fake_generated):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.criterion(real_target, fake_generated, phase='train')
        loss.backward()
        self.optimizer.step()

        return loss

    def eval_step(self, real_target, fake_generated):
        self.model.eval()
        with torch.no_grad():
            _ = self.criterion(real_target, fake_generated, phase='valid')
    
    def step_scheduler(self, val_loss):
        self.scheduler.step(val_loss)
    
    def reset_losses(self):
        self.losses_history = {'train': [], 'valid': []}
    
    def save_model_state_dict(self, output_path):
        torch.save(self.model.state_dict(), os.path.join(output_path, "critic.pt"))

    def load_model_state_dict(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            print(f"Загружено состояние критика из: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Файл {checkpoint_path} не найден.")
