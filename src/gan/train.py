import os
import torch
import matplotlib

from src.common.structs import ExecPhase
from src.common.interfaces import IGenerativeModel
from src.gan.dataset import DatasetCreator
from src.gan.custom_evaluators import *


matplotlib.use('Agg')
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, 
                 device: torch.device,
                 generative_model: IGenerativeModel,
                 dataset_processor: DatasetCreator, 
                 output_path: str, 
                 epochs: int, 
                 batch_size: int, 
                 load_weights: bool=False, 
                 validation_ratio: float=0.2, 
                 checkpoints_freq: int=25):
        self.device = device
        self.generative_model = generative_model
        self.dataset_processor = dataset_processor
        self.output_path = output_path
        self.epochs = epochs
        self.val_ratio = validation_ratio
        self.batch_size = batch_size
        self.checkpoints_ratio = checkpoints_freq
        self.load_weights = load_weights
        
        self.t_loader, self.v_loader = self.dataset_processor.get_dataloaders(
            batch_size=self.batch_size, 
            shuffle=True, 
            workers=6, 
            val_ratio=self.val_ratio
        )
        os.makedirs(self.output_path, exist_ok=True)
    
    def train(self):
        # реализовать подгрузку чекпоинта
        if self.load_weights:
            pass
        
        for epoch in range(self.epochs):
            print(f"\nЭпоха {epoch + 1}/{self.epochs}")

            for phase, loader in [(ExecPhase.TRAIN, self.t_loader), (ExecPhase.VALID, self.v_loader)]:
                for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} {phase.value.capitalize()}")):
                    inp, trg = [el.to(self.device) for el in batch]
                    
                    self.generative_model.model_step(inp, trg, phase)
                    if i == 0:
                        self.generative_model.model_step(inp, trg, ExecPhase.EVAL)

            # расчёт и вывод средних лоссов по эпохе
            print(f'===========[{ExecPhase.TRAIN.value}] EVALUATORS SUMMARY')
            self.generative_model.print_eval_summary(phase=ExecPhase.TRAIN)
            
            print(f'===========[{ExecPhase.VALID.value}] EVALUATORS SUMMARY')
            self.generative_model.print_eval_summary(phase=ExecPhase.VALID)
            
            self.generative_model.clear_eval_history()
