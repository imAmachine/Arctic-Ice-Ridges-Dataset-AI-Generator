import os
import re
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

class ParamGridTester:
    def __init__(self, param_grid, output_root=None, seed=42):
        self.param_grid = param_grid
        self.seed = seed
        self.output_path = None
        self.trainer = GANTrainer
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
        folder_name = self._create_output_path(params)
     
        self.trainer = self._get_new_trainer(params)
        self.trainer.train()

        Utils.to_json(data=params, path=os.path.join(self.output_path, "config.json"))
        
        self._append_summary(params, self.trainer, folder_name)
        self._save_test(self.trainer.model, self.trainer.loaders)
    
    def _save_test(self, gen_model: 'GenerativeModel', loaders: Dict) -> None:
        Utils.to_json(data=self.trainer.metrics_history, path=os.path.join(self.output_path, 'metrics_history.json'))
        Utils.to_json({
            'generator': gen_model.g_trainer.losses_history,
            'critic': gen_model.c_trainer.losses_history
        }, os.path.join(self.output_path, 'losses_history.json'))
        
        for phase, loader in loaders.items():
            final_batch = next(iter(loader))
            self.trainer._process_batch(phase=phase, batch=final_batch, visualize_batch=True)
    
    def _fix_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _generate_combinations(self) -> List[Dict]:

        flat_params = self._flatten_dict("", self.param_grid)
        keys, values = zip(*flat_params.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        nested_combinations = [self._unflatten_dict(combo) for combo in combinations]

        return nested_combinations
    
    def _flatten_dict(self, prefix: str, params: Dict) -> Dict[str, list]:
            flat = {}
            for name, value in params.items():
                full_key = f"{prefix}.{name}" if prefix else name
                if isinstance(value, dict):
                    flat.update(self._flatten_dict(full_key, value))
                else:
                    flat[full_key] = value if isinstance(value, list) else [value]
            return flat
    
    def _unflatten_dict(self, flat_dict: Dict) -> Dict:
        nested = {}
        for name, value in flat_dict.items():
            parts = name.split('.')
            current_level = nested
            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            current_level[parts[-1]] = value
        return nested

    def _create_output_paths(self, params) -> str:
        gen_params = params.get('GenerativeModel', {})
        if not gen_params:
            raise ValueError("params['GenerativeModel'] is missing or empty.")

        flat = []
        for name, value in gen_params.items():
            if isinstance(value, dict):
                for sub_k, sub_v in value.items():
                    flat.append(f'{self.remove_symbols(sub_k)}_{self.remove_symbols(sub_v)}')
            else:
                flat.append(f'{self.remove_symbols(name)}_{self.remove_symbols(value)}')
        return '_'.join(flat)
    
    def remove_symbols(self, val):
        val = str(val)
        val = re.sub(r'[^a-zA-Z0-9\-_]', '', val)
        return val

    def _get_new_trainer(self, params: Dict) -> 'GANTrainer':
        model = GenerativeModel(
            **params.get('GenerativeModel'),
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
            augs_per_img=params.get('augs_per_img', 1),
            device=DEVICE
        )

        return GANTrainer(
            model=model,
            **params.get('GANTrainer'),
            dataset_processor=ds_creator,
            output_path=self.output_path,
            device=DEVICE,
            load_weights=False,
            checkpoints_ratio=5
        )
    
    def _create_output_path(self, params) -> tuple[str, str]:
        folder_name = self._create_output_paths(params)
        self.output_path = os.path.join(self.output_root, f"grid_{folder_name}")
        os.makedirs(self.output_path, exist_ok=True)
        return folder_name
    
    def _append_summary(self, params, trainer: 'GANTrainer', folder_name: str):
        summary_row = {'folder': folder_name, **params}
        
        val_metrics_res = {name: metric[-1] for name, metric in trainer.metrics_history['valid'].items() if len(metric) > 0}
        summary_row.update(val_metrics_res)

        df = pd.DataFrame([summary_row])
        if os.path.exists(self.results_summary_path):
            df.to_csv(self.results_summary_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_summary_path, mode='w', header=True, index=False)