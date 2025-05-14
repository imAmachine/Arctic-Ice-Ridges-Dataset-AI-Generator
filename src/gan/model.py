from copy import copy
import os
from typing import Dict, List
import torch
import torch.nn as nn

from src.gan.models_evaluating import Evaluator, EvalProcessor
from src.gan.custom_evaluators import *
from src.common.interfaces import IGenerativeModel
from src.gan.arch import WGanCritic, WGanGenerator
from src.common.structs import ExecPhase as phases, ModelType as models, LossName as losses, MetricsName as metrics, EvaluatorType as eval_type


class ModelTrainer:
    def __init__(self, model, optimizer, scheduler, evaluate_processor: EvalProcessor):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluate_processor: EvalProcessor = evaluate_processor
    
    def process_losses(self, generated_sample, real_sample, history_key: phases) -> 'torch.Tensor':
        self.evaluate_processor.process(generated_sample=generated_sample,
                                        real_sample=real_sample,
                                        exec_phase=history_key)
        
        processed_losses = self.evaluate_processor.evaluators_history[history_key][-1][eval_type.LOSS.value].values()
        
        return sum(processed_losses)
    
    def optimization_step(self, real_sample, generated_sample) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # подсчёт эвалуаторов и градиентный шаг
        loss_tensor = self.process_losses(generated_sample, real_sample, history_key=phases.TRAIN)
        loss_tensor.backward()
        
        self.optimizer.step()
        
        return loss_tensor.item()
    
    def evaluation_step(self, real_sample, generated_sample) -> float:
        self.model.eval()
        
        loss_tensor = self.process_losses(generated_sample=generated_sample,
                                          real_sample=real_sample,
                                          history_key=phases.VALID)
        
        return loss_tensor.item()
    
    def get_summary(self, name, phase: phases):
        summary = self.evaluate_processor.compute_epoch_summary(phase=phase)

        for group, metrics in summary.items():
            print('\n')
            if len(metrics.items()) > 0:
                print(f"[{name}] {group.upper()}:")
                for k, v in metrics.items():
                    print(f"\t{k}: {v:.4f}")

        self.evaluate_processor.reset_history()
            


class GenerativeModel(IGenerativeModel):
    def __init__(self, device: str, 
                 evaluators_info: Dict,
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
        g_eval_info, d_eval_info = evaluators_info.get(models.GENERATOR.value), evaluators_info.get(models.DISCRIMINATOR.value)
        
        self.evaluation_funcs = {
            losses.ADVERSARIAL.value: AdversarialLoss(self.discriminator),
            losses.BCE.value: nn.BCELoss(),
            losses.L1.value: nn.L1Loss(),
            losses.WASSERSTEIN.value: WassersteinLoss(self.discriminator),
            losses.GP.value: GradientPenalty(self.discriminator),
            
            metrics.PRECISION.value: sklearn_wrapper(precision_score, device),
            metrics.F1.value: sklearn_wrapper(f1_score, device),
            metrics.IOU.value: sklearn_wrapper(jaccard_score, device),
        }

        self.g_trainer = self.init_trainer(self.generator, g_eval_info)
        self.d_trainer = self.init_trainer(self.discriminator, d_eval_info)
    
    def build_evaluators(self, funcs, evaluators_info):
        return [Evaluator(callable_fn=funcs[k], name=k, **v) for k, v in evaluators_info.items()]
    
    def init_trainer(self, model, eval_info: Dict):
        optimizers = [torch.optim.RMSprop(model.parameters(), lr=self.optimization_params.get('lr')),
                      torch.optim.Adam(model.parameters(), lr=self.optimization_params.get('lr'), betas=(0.0, 0.9))]
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[1], mode=self.optimization_params.get('mode'), factor=0.5, patience=6)
        
        model_evaluators = self.build_evaluators(self.evaluation_funcs, eval_info)
        
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizers[1],
            scheduler=scheduler,
            evaluate_processor=EvalProcessor(device=self.device, evaluators=model_evaluators)
        )
        
        return trainer

    def switch_phase(self, phase: phases=phases.TRAIN) -> None:
        if phase == phases.TRAIN:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()
    
    def __train_critic_step(self, input_data, target_data, inpaint_mask):
        for _ in range(self.n_critic):
            with torch.no_grad():
                generated = self.generator(input_data, inpaint_mask)
            self.d_trainer.optimization_step(target_data, generated)
    
    def __train_generator_step(self, input_data, target_data, inpaint_mask):
        generated = self.generator(input_data, inpaint_mask)
        self.g_trainer.optimization_step(target_data, generated)
    
    def train_step(self, batch: tuple[torch.Tensor,]) -> None:
        self.current_iteration += 1
        
        # цикл обучения критика
        self.__train_critic_step(*batch)
        
        # шаг обучения генератора
        self.__train_generator_step(*batch)

    def valid_step(self, batch: tuple[torch.Tensor,]) -> None:
        input_data, target_data, inpaint_mask = batch
        
        with torch.no_grad():
            generated = self.g_trainer.model(input_data, inpaint_mask)
            
            self.g_trainer.evaluation_step(target_data, generated)
            self.d_trainer.evaluation_step(target_data, generated)

    # def infer_generate(self, preprocessed_img, checkpoint_path, processor): # НУЖНО РАЗГРЕСТИ ЭТОТ МУСОР
    #     self.switch_phase(phases.EVAL)
    #     self.load_checkpoint(checkpoint_path)

    #     damaged, original, outpaint_mask = IceRidgeDataset.prepare_data(
    #         img=preprocessed_img,
    #         processor=processor,
    #         augmentations=None,
    #         model_transforms=self.generator.get_model_transforms(self.target_image_size)
    #     )
    #     damaged = damaged.to(self.device)
    #     outpaint_mask = outpaint_mask.to(self.device)

    #     with torch.no_grad():
    #         generated = self.generator(damaged.unsqueeze(1), outpaint_mask.unsqueeze(1))

    #     generated_img = generated.detach().cpu().squeeze().numpy() * 255
    #     original_img = original.squeeze().numpy() * 255

    #     return generated_img, original_img
