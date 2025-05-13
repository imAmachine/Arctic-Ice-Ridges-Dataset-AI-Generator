import os
from typing import Dict, List
import torch
import torch.nn as nn

from src.gan.loss_processing import Loss, LossProcessor
from src.gan.custom_losses import *
from src.gan.dataset import IceRidgeDataset
from src.common.interfaces import IGenerativeModel
from src.gan.arch import WGanCritic, WGanGenerator
from src.common.structs import ExecPhases as phases, ModelTypes as models, LossNames as losses


class ModelTrainer:
    def __init__(self, model, device: str, optimizer = None, scheduler = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.loss_processor = LossProcessor(device)
    
    def train_step(self, real_sample, generated_sample) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.loss_processor.calc_losses(real_sample, generated_sample, phase=phases.TRAIN)
        loss.backward()
        
        self.optimizer.step()
    
    def eval_step(self, real_sample, generated_sample) -> None:
        self.model.eval()
        _ = self.loss_processor.calc_losses(real_sample, generated_sample, phase=phases.VALID)
            
    def step_scheduler(self, metric):
        self.scheduler.step(metric)


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
        self.discriminator = WGanCritic(input_channels=1, feature_maps=d_feature_maps).to(self.device)
        self.current_iteration = 0

        self.n_critic = n_critic
        
        g_losses_weights, d_losses_weights = losses_weights.get(models.GENERATOR.value), losses_weights.get(models.DISCRIMINATOR.value)
        
        g_losses_list = [
            Loss(loss_fn=AdversarialLoss(self.discriminator), name=losses.ADVERSARIAL.value, weight=g_losses_weights.get(losses.ADVERSARIAL.value)),
            Loss(loss_fn=nn.BCELoss(), name=losses.BCE.value, weight=g_losses_weights.get(losses.BCE.value)),
            Loss(loss_fn=nn.L1Loss(), name=losses.L1.value, weight=g_losses_weights.get(losses.L1.value)),
        ]
        
        d_losses_list = [
            Loss(loss_fn=WassersteinLoss(self.discriminator), name=losses.WASSERSTEIN.value, weight=d_losses_weights.get(losses.WASSERSTEIN.value)),
            Loss(loss_fn=GradientPenalty(self.discriminator), name=losses.GP.value, weight=d_losses_weights.get(losses.GP.value), only_on=phases.TRAIN),
        ]
        
        self.g_trainer = self._init_trainer(self.generator, g_losses_list)
        self.d_trainer = self._init_trainer(self.discriminator, d_losses_list)
    
    def _init_trainer(self, model, losses_list: List['Loss']):
        optimizers = [torch.optim.RMSprop(model.parameters(), lr=self.optimization_params.get('lr')),
                      torch.optim.Adam(model.parameters(), lr=self.optimization_params.get('lr'), betas=(0.0, 0.9))]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[1], mode=self.optimization_params.get('mode'), factor=0.5, patience=6)
        
        trainer = ModelTrainer(
            model=model,
            device=self.device,
            optimizer=optimizers[1],
            scheduler=scheduler
        )
        
        trainer.loss_processor.new_losses(losses_list)
        
        return trainer

    def switch_phase(self, phase: phases=phases.TRAIN) -> None:
        if phase == phases.TRAIN:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()
    
    def _train_critic_step(self, input_data, target_data, inpaint_mask):
        for _ in range(self.n_critic):
            with torch.no_grad():
                generated = self.generator(input_data, inpaint_mask)
            self.d_trainer.train_step(target_data, generated)
            
        # real_score = self.discriminator(target_data).mean().item()
        # fake_score = self.discriminator(generated).mean().item()
        # print(f"D(real)={real_score:.4f}, D(fake)={fake_score:.4f}")
    
    def _train_generator_step(self, input_data, target_data, inpaint_mask):
        generated = self.generator(input_data, inpaint_mask)
        self.g_trainer.train_step(target_data, generated)
    
    def train_step(self, batch: tuple[torch.Tensor,]) -> None:
        self.current_iteration += 1
        
        # цикл обучения критика
        self._train_critic_step(*batch)
        
        # шаг обучения генератора
        self._train_generator_step(*batch)

    def valid_step(self, batch: tuple[torch.Tensor,]) -> None:
        input_data, target_data, inpaint_mask = batch
        
        with torch.no_grad():
            generated = self.g_trainer.model(input_data, inpaint_mask)
            
            self.g_trainer.eval_step(target_data, generated)
            self.d_trainer.eval_step(target_data, generated)

    def step_schedulers(self, metric: float) -> None:
        self.g_trainer.step_scheduler(metric)
        self.d_trainer.step_scheduler(metric)

    def save_checkpoint(self, output_path): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
        os.makedirs(output_path, exist_ok=True)
        checkpoint = {
            'current_iteration': self.current_iteration,
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.discriminator.state_dict(),
            'generator_optimizer': self.g_trainer.optimizer.state_dict(),
            'critic_optimizer': self.d_trainer.optimizer.state_dict(),
            'generator_scheduler': self.g_trainer.scheduler.state_dict(),
            'critic_scheduler': self.d_trainer.scheduler.state_dict(),
            'generator_loss_history': self.g_trainer.loss_processor.loss_history,
            'critic_loss_history': self.d_trainer.loss_processor.loss_history
        }
        torch.save(checkpoint, os.path.join(output_path, 'training_checkpoint.pt'))
        print(f"Checkpoint сохранен в {os.path.join(output_path, 'training_checkpoint.pt')}")

    def load_checkpoint(self, path): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
        checkpoint_path = os.path.join(path, 'training_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['critic_state_dict'])

            self.g_trainer.optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.d_trainer.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.g_trainer.scheduler.load_state_dict(checkpoint['generator_scheduler'])
            self.d_trainer.scheduler.load_state_dict(checkpoint['critic_scheduler'])

            self.g_trainer.loss_history = checkpoint['generator_loss_history']
            self.d_trainer.loss_history = checkpoint['critic_loss_history']
            self.current_iteration = checkpoint['current_iteration']

            print(f"Checkpoint загружен из {checkpoint_path} (итерация {self.current_iteration})")
        else:
            raise FileNotFoundError(f"Checkpoint файл не найден по пути {checkpoint_path}")

    def infer_generate(self, preprocessed_img, checkpoint_path, processor): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
        self.switch_phase(phases.EVAL)
        self.load_checkpoint(checkpoint_path)

        damaged, original, outpaint_mask = IceRidgeDataset.prepare_data(
            img=preprocessed_img,
            processor=processor,
            augmentations=None,
            model_transforms=self.generator.get_model_transforms(self.target_image_size)
        )
        damaged = damaged.to(self.device)
        outpaint_mask = outpaint_mask.to(self.device)

        with torch.no_grad():
            generated = self.generator(damaged.unsqueeze(1), outpaint_mask.unsqueeze(1))

        generated_img = generated.detach().cpu().squeeze().numpy() * 255
        original_img = original.squeeze().numpy() * 255

        return generated_img, original_img
