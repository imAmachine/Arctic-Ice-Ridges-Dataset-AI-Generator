import os
import random
import itertools
import numpy as np
import torch
import pandas as pd

from settings import *
from typing import List, Dict

from src.common.utils import Utils
from src.gan.model import GenerativeModel
from src.gan.train import GANTrainer
from src.gan.dataset import DatasetCreator
from src.gan.arch import AUGMENTATIONS

# Нужно доработать класс!
class ParamGridTester:
    def __init__(self, param_grid, output_root=None, seed=42):
        self.param_grid = param_grid
        self.seed = seed
        self.combinations = self._generate_combinations()
        self.output_root = output_root or os.path.join(WEIGHTS_PATH, 'test')
        self.results_summary_path = os.path.join(self.output_root, 'results.csv')
        self._fix_seed()

    def run_grid_tests(self):
        os.makedirs(self.output_root, exist_ok=True)
        for i, params in enumerate(self.combinations):
            print(f"\n=== [{i+1}/{len(self.combinations)}] ===\nParams: {params}")
            self._grid_tester_iter(params)
        print(f'Результаты тестов сохранены в {self.results_summary_path}')
    
    def _grid_tester_iter(self, params):
        folder_name, output_path = self._create_output_path()
        
        trainer = self._get_new_trainer(output_path)
        trainer.train()

        Utils.to_json(data=params, path=os.path.join(output_path, "config.json"))
        Utils.to_json(data=trainer.metrics_history, path=os.path.join(output_path, 'metrics_history.json'))
        
        self._append_summary(params, trainer, folder_name)
        self._save_test(trainer.loaders)
    
    def _save_test(self, gen_model: 'GenerativeModel', loaders: Dict):
        Utils.to_json(self.metrics_history, os.path.join(self.output_path, 'metrics_history.json'))
        Utils.to_json({
            'generator': gen_model.g_trainer.losses_history,
            'critic': gen_model.c_trainer.losses_history
        }, os.path.join(self.output_path, 'losses_history.json'))
        
        for phase, loader in loaders.items():
            final_batch = next(iter(loader))
            self._process_batch(phase=phase, batch=final_batch, visualize_batch=True)
    
    def _fix_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _generate_combinations(self) -> List[Dict]:
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _create_output_paths(self) -> None:
        return '\n'.join([f'{k}_{v}-' for k, v in self.param_grid.get('GenerativeModel').items()])

    def _get_new_trainer(self, output_path: str) -> 'GANTrainer':
        model = GenerativeModel(
            **self.param_grid.get('GenerativeModel'),
            device=DEVICE
        )

        ds_creator = DatasetCreator(
            generated_path=AUGMENTED_DATASET_FOLDER_PATH,
            original_data_path=MASKS_FOLDER_PATH,
            preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
            images_extentions=MASKS_FILE_EXTENSIONS,
            model_transforms=model.get_model_transforms(),
            preprocessors=PREPROCESSORS,
            augmentations=AUGMENTATIONS,
            device=DEVICE
        )

        return GANTrainer(
            model=model,
            **self.param_grid.get('GANTrainer'),
            dataset_processor=ds_creator,
            output_path=output_path,
            device=DEVICE,
            load_weights=False,
            checkpoints_ratio=5
        )
    
    def _create_output_path(self) -> tuple[str, str]:
        folder_name = self._create_output_paths()
        output_path = os.path.join(self.output_root, f"grid_{folder_name}")
        os.makedirs(output_path, exist_ok=True)
        return folder_name, output_path
    
    def _append_summary(self, params, trainer: 'GANTrainer', folder_name: str):
        summary_row = {'folder': folder_name, **params}
        
        val_metrics_res = {name: metric[-1] for name, metric in trainer.metrics_history['valid'].items() if len(metric) > 0}
        summary_row.update(val_metrics_res)

        df = pd.DataFrame([summary_row])
        if os.path.exists(self.results_summary_path):
            df.to_csv(self.results_summary_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_summary_path, mode='w', header=True, index=False)