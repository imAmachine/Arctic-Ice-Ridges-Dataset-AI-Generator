import os
import random
import itertools
import numpy as np
import torch
import pandas as pd

from settings import *
from typing import List, Dict

from src.common.utils import Utils
from src.gan.model import GenerativeModel_GAN
from src.gan.train import GANTrainer
from src.gan.dataset import DatasetCreator
from src.gan.arch import AUGMENTATIONS


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
        for i, cfg in enumerate(self.combinations):
            print(f"\n=== [{i+1}/{len(self.combinations)}] ===\nParams: {cfg}")
            self._grid_tester_iter(cfg)
        print(f'Результаты тестов сохранены в {self.results_summary_path}')
    
    def _grid_tester_iter(self, cfg: List[Dict]):
        folder_name, output_path = self._create_output_path(cfg)
        
        trainer = self._get_new_trainer(cfg, output_path)
        trainer.train()

        Utils.to_json(data=cfg, path=os.path.join(output_path, "config.json"))
        Utils.to_json(data=trainer.metrics_history, path=os.path.join(output_path, 'metrics_history.json'))
        
        self._append_summary(cfg, trainer, folder_name)
    
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

    def _create_output_paths(self, cfg: List[Dict]) -> None:
        return (
            f"g{cfg['g_feature_maps']}_d{cfg['d_feature_maps']}"
            f"_w{cfg['lambda_w']}_bce{cfg['lambda_bce']}"
            f"_l1{cfg['lambda_l1']}_gp{cfg['lambda_gp']}"
            f"_nc{cfg['n_critic']}_ep{cfg['epochs']}"
        )

    def _get_new_trainer(self, cfg: List[Dict], output_path: str) -> 'GANTrainer':
        model = GenerativeModel_GAN(
            target_image_size=cfg['target_image_size'],
            g_feature_maps=cfg['g_feature_maps'],
            d_feature_maps=cfg['d_feature_maps'],
            device=DEVICE,
            lr=cfg['lr'],
            n_critic=cfg['n_critic'],
            lambda_w=cfg['lambda_w'],
            lambda_bce=cfg['lambda_bce'],
            lambda_gp=cfg['lambda_gp'],
            lambda_l1=cfg['lambda_l1']
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
            dataset_processor=ds_creator,
            output_path=output_path,
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            device=DEVICE,
            load_weights=False,
            val_ratio=cfg['val_ratio'],
            checkpoints_ratio=5
        )
    
    def _create_output_path(self, cfg: List[Dict]) -> tuple[str, str]:
        folder_name = self._create_output_paths(cfg)
        output_path = os.path.join(self.output_root, f"grid_{folder_name}")
        os.makedirs(output_path, exist_ok=True)
        return folder_name, output_path
    
    def _append_summary(self, cfg, trainer: 'GANTrainer', folder_name: str):
        summary_row = {'folder': folder_name, **cfg}
        
        val_metrics_res = {name: metric[-1] for name, metric in trainer.metrics_history['valid'].items() if len(metric) > 0}
        summary_row.update(val_metrics_res)

        df = pd.DataFrame([summary_row])
        if os.path.exists(self.results_summary_path):
            df.to_csv(self.results_summary_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_summary_path, mode='w', header=True, index=False)