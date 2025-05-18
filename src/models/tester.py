# import os
# import re
# import random
# import itertools
# import numpy as np
# import torch
# import pandas as pd

# from config.settings import *
# from typing import List, Dict

# from src.common.utils import Utils
# from dataset.loader import DatasetCreator
# from src.models.train import GAN, Trainer

# class ParamGridTester:
#     def __init__(self, param_grid_config, trainer: Trainer, output_folder_path, seed=42):
#         self.param_grid = param_grid_config
#         self.trainer = trainer
#         self.output_folder_path = os.path.join(output_folder_path, 'test')
#         self.results_summary_path = os.path.join(self.output_folder_path, 'results.csv')
        
#         self.combinations = self._generate_combinations()
#         self._fix_seed(seed)

#     def run_grid_tests(self):
#         os.makedirs(self.output_root, exist_ok=True)
#         for i, params in enumerate(self.combinations):
#             print(f"\n=== [{i+1}/{len(self.combinations)}] ===\nParams: {params}")
#             folder_name = self._create_output_path(i)
#             self._grid_tester_iter(params, folder_name)
#         print(f'Результаты тестов сохранены в {self.results_summary_path}')
    
#     def _grid_tester_iter(self, params: Dict, folder_name):
#         self.trainer = self._get_new_trainer(params)
#         self.trainer.run()

#         Utils.to_json(data=params, path=os.path.join(self.output_folder_path, "config.json"))
        
#         self._append_summary(params, self.trainer, folder_name)
#         self._save_test(self.trainer.model, self.trainer.val_loader)
    
#     def _save_test(self, gen_model: 'GAN', loaders: Dict) -> None:
#         Utils.to_json(
#             self.trainer.metrics_history, 
#             os.path.join(self.output_folder_path, 'metrics_history.json')
#         )
#         Utils.to_json(
#             self.trainer.losses_history, 
#             os.path.join(self.output_folder_path, 'losses_history.json')
#         )
    
#     def _fix_seed(self, seed) -> None:
#         random.seed(self.seed)
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         torch.cuda.manual_seed_all(self.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     def _generate_combinations(self) -> List[Dict]:
#         flat_params = self._flatten_dict("", self.param_grid)
#         keys, values = zip(*flat_params.items())
#         combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
#         nested_combinations = [self._unflatten_dict(combo) for combo in combinations]

#         return nested_combinations
    
#     def _flatten_dict(self, prefix: str, params: Dict) -> Dict[str, list]:
#         flat = {}
#         for name, value in params.items():
#             full_key = f"{prefix}.{name}" if prefix else name
#             if isinstance(value, dict):
#                 flat.update(self._flatten_dict(full_key, value))
#             else:
#                 flat[full_key] = value if isinstance(value, list) else [value]
#         return flat
    
#     def _unflatten_dict(self, flat_dict: Dict) -> Dict:
#         nested = {}
#         for name, value in flat_dict.items():
#             parts = name.split('.')
#             current_level = nested
#             for part in parts[:-1]:
#                 current_level = current_level.setdefault(part, {})
#             current_level[parts[-1]] = value
#         return nested
    
#     def remove_symbols(self, val):
#         val = str(val)
#         val = re.sub(r'[^a-zA-Z0-9\-_]', '', val)
#         return val

#     def _get_new_trainer(self, params: Dict) -> 'Trainer':
#         modules = GAN._init_modules(params) 
#         model = GAN(DEVICE, modules, n_critic=params["n_critic"])

#         ds_creator = DatasetCreator(
#             input_preprocessor=self.dataset_preprocessor,
#             masking_processor=self.masking_processor,
#             model_transforms=self.transforms,
#             augs_per_img=params["Dataset"]["augs_per_img"]
#         )

#         return Trainer(
#             device=DEVICE,
#             model=model,
#             dataset=ds_creator,
#             output_path=self.output_folder_path,
#             epochs=params["Trainer"]["epochs"],
#             batch_size=params["Trainer"]["batch_size"],
#             val_ratio=params["Trainer"]["val_ratio"]
#         )
    
#     def _create_output_path(self, index: int) -> str:
#         folder_name = f"grid_{index+1}"
#         self.output_folder_path = os.path.join(self.output_root, folder_name)
#         os.makedirs(self.output_folder_path, exist_ok=True)
#         return folder_name
    
#     def _append_summary(self, params, trainer: 'Trainer', folder_name: str):
#         summary_row = {'folder': folder_name, **params}

#         last_metrics = trainer.metrics_history[-1]
#         print(f'last_metrics: {last_metrics}')
        
#         for model_type, mgr in trainer.model.trainers.items():
#             model_metrics = {
#                 key: value[model_type.value]
#                 for key, value in last_metrics.items()
#                 if (
#                     isinstance(value, dict)
#                     and model_type.value in value
#                     and key.startswith("Metric")
#                     and value[model_type.value] is not None
#                     and not (isinstance(value[model_type.value], float) and np.isnan(value[model_type.value]))
#                 )
#             }
#             print(f'model_metrics: {model_metrics}')
            
#             for metric_name, value in model_metrics.items():
#                 key = f"{metric_name}"
#                 summary_row[key] = value

#         df_out = pd.DataFrame([summary_row])
#         if os.path.exists(self.results_summary_path):
#             df_out.to_csv(self.results_summary_path, mode='a', header=False, index=False)
#         else:
#             df_out.to_csv(self.results_summary_path, mode='w', header=True, index=False)