import os
from typing import Dict, Literal
import torch
import torch.nn as nn
from torchvision.transforms import transforms, InterpolationMode

from src.gan.custom_losses import adversarial_loss, gradient_penalty, wasserstein_loss
from src.gan.dataset import IceRidgeDataset
from src.common.interfaces import IGenerativeModel, IModelTrainer
from src.gan.arch import WGanCritic, WGanGenerator
from src.common.structs import TrainPhases as phases


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
            device=self.device,
            critic=self.critic,
            losses_weights=g_losses_weights
        )
        
        c_trainer = WGANCriticModelTrainer(
            model=self.critic,
            device=self.device,
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

    def switch_mode(self, mode: phases=phases.TRAIN) -> None:
        self.generator.train() if mode == 'train' else self.generator.eval()
        self.critic.train() if mode == 'train' else self.critic.eval()
    
    def _train_critic_step(self, input_data, target_data, damage_mask):
        for _ in range(self.n_critic):
            with torch.no_grad():
                fake_images = self.generator(input_data, damage_mask)
            self.c_trainer.train_step(samples=(target_data, fake_images,))
    
    def _train_generator_step(self, input_data, target_data, damage_mask):
        generated = self.generator(input_data, damage_mask)
        self.g_trainer.train_step(samples=(generated, target_data,))
    
    def train_step(self, batch: tuple[torch.Tensor,]) -> None:
        input_data, target_data, inpaint_mask = batch
        self.current_iteration += 1
        
        # цикл обучения критика
        self._train_critic_step(input_data, target_data, inpaint_mask)
        
        # шаг обучения генератора
        self._train_generator_step(input_data, target_data, inpaint_mask)

    def valid_step(self, batch: tuple[torch.Tensor,]) -> None:
        input_data, target_data, inpaint_mask = batch
        
        with torch.no_grad():
            self.g_trainer.eval_step(samples=(input_data, target_data,))
            generated = self.g_trainer.model(input_data, inpaint_mask)
            
            self.c_trainer.eval_step(samples=(target_data, generated,))

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
    def __init__(self, model, device, critic, losses_weights: Dict):
        super().__init__(model, device, losses_weights)
        self.critic = critic
        self.criterion = self._losses
    
    def _losses(self, input_data, target_data, phase: phases=phases.TRAIN) -> None:
        self.calc_loss(loss_fn=adversarial_loss, loss_name='adv', phase=phase, args=(self.critic, input_data,))
        self.calc_loss(loss_fn=nn.BCELoss(), loss_name='bce', phase=phase, args=(input_data, target_data))


class WGANCriticModelTrainer(IModelTrainer):
    def __init__(self, model, device, losses_weights: Dict):
        super().__init__(model, device, losses_weights)
    
    def _losses(self, real_target, fake_generated, phase: phases=phases.TRAIN):
        self.calc_loss(loss_fn=wasserstein_loss, loss_name='wasserstein', phase=phase, args=(self.model, real_target, fake_generated))
        
        if phase == phases.TRAIN:
            self.calc_loss(loss_fn=gradient_penalty, loss_name='gp', phase=phase, args=(self.model, self.device, real_target, fake_generated))    
