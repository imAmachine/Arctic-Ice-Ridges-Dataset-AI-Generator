import os
import random
import numpy as np
import torch

from config.path import *
from typing import Dict

from src.common.utils import Utils
from src.dataset.loader import DatasetCreator
from src.models.csv_saver import TestResultSaver
from src.models.train import Trainer
from src.models.param_combination import ParamGridCombination
        
class ParamGridTester:
    def __init__(self, param_grid_config: Dict, trainer: Trainer, dataset: DatasetCreator, output_folder_path: str, seed: int = 42):
        self.output_folder_path = os.path.join(output_folder_path, 'test')

        self.trainer = trainer
        self.dataset = dataset
        self.result_logger = TestResultSaver(os.path.join(self.output_folder_path, 'results.csv'))
        self.param_generator = ParamGridCombination(param_grid_config)
        self.param_combinations = self.param_generator.generate_combination()

        self._fix_seed(seed)

    def run(self):
        for i, params in enumerate(self.param_combinations):
            print(f"\n=== [{i+1}/{len(self.param_combinations)}] ===\nParams: {params}")
            folder = self._create_output_path(i)
            self._reinitialize_components(params, folder)
            self._create_config_json(params, folder)
            self.trainer.run()
            self.result_logger.save(folder, params, self.trainer)

    def _create_config_json(self, params, folder_path):
        Utils.to_json(data=params, path=os.path.join(folder_path, "config.json"))

    def _fix_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_output_path(self, index: int) -> str:
        folder_path = os.path.join(self.output_folder_path, f"grid_{index+1}")
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def _reinitialize_components(self, params: Dict, folder_path: str) -> None:
        self.trainer.model.build(params)

        self.dataset.__init__(
            metadata=self.dataset.metadata,
            mask_processor=self.dataset.mask_processor,
            transforms=self.dataset.transforms,
            augs_per_img=params['Dataset']['augs_per_img'],
            valid_size_p=params['Dataset']['val_ratio'],
            shuffle=self.dataset.shuffle,
            batch_size=params['Dataset']['batch_size'],
            workers=params['Dataset']['workers']
        )

        self.trainer.__init__(
            device=self.trainer.device,
            model=self.trainer.model,
            dataloaders=self.dataset.create_loaders(),
            output_path=folder_path,
            epochs=params['Trainer']['epochs'],
            checkpoints_ratio=params['Trainer']['checkpoints_ratio']
            )