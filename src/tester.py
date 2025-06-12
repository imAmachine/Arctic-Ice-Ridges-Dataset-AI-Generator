### НЕОБХОДИМО ПЕРЕРАБОТАТЬ!!!!!!!!!!!!!!!

# from __future__ import annotations
# import itertools
# import os
# import random
# import numpy as np
# import torch

# from typing import Dict, List

# from generativelib.common.utils import Utils
# from generativelib.dataset.loader import DatasetCreator
# from generativelib.model.train.train import TrainManager


# class ParamGridCombination:
#     def __init__(self, grid_config: Dict):
#         self.grid_config = grid_config

#     def generate_combination(self) -> List[Dict]:
#         flat_params = self._flatten("", self.grid_config)
#         keys, values = zip(*flat_params.items())
#         combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
#         return [self._unflatten(combo) for combo in combinations]

#     def _flatten(self, prefix: str, params: Dict) -> Dict[str, List]:
#         flat = {}
#         for name, value in params.items():
#             full_key = f"{prefix}.{name}" if prefix else name
#             if isinstance(value, dict):
#                 flat.update(self._flatten(full_key, value))
#             else:
#                 flat[full_key] = value if isinstance(value, list) else [value]
#         return flat

#     def _unflatten(self, flat: Dict) -> Dict:
#         nested = {}
#         for compound_key, value in flat.items():
#             keys = compound_key.split('.')
#             current_level = nested
#             for key in keys[:-1]:
#                 current_level = current_level.setdefault(key, {})
#             current_level[keys[-1]] = value
#         return nested


# class ParamGridTester:
#     def __init__(self, param_grid_config: Dict, trainer: TrainManager, dataset: DatasetCreator, output_folder_path: str, seed: int = 42):
#         self.trainer = trainer
#         self.dataset = dataset
#         self.param_generator = ParamGridCombination(param_grid_config)
#         self.param_combinations = self.param_generator.generate_combination()

#         self._fix_seed(seed)
#         self.output_folder_path = os.path.join(output_folder_path, 'tests')

#     def run(self):
#         for i, params in enumerate(self.param_combinations):
#             print(f"\n=== [{i+1}/{len(self.param_combinations)}] ===  ТЕСТИРОВАНИЕ")
#             folder = self._create_output_path(i)
#             result_path = os.path.join(self.output_folder_path, 'results.csv')
            
#             self._reinitialize_components(params, folder)
#             self._create_config_json(params, folder)
#             self.trainer.run()
            
#             self.trainer.model.evaluators_summary(result_path)

#     def _create_config_json(self, params, folder_path):
#         Utils.to_json(data=params, path=os.path.join(folder_path, "config.json"))

#     def _fix_seed(self, seed: int) -> None:
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     def _create_output_path(self, index: int) -> str:
#         folder_path = os.path.join(self.output_folder_path, f"grid_{index+1}")
#         os.makedirs(folder_path, exist_ok=True)
#         return folder_path
    
#     def _reinitialize_components(self, params: Dict, folder_path: str) -> None:
#         self.trainer.model.build_train_modules(params)

#         self.dataset.__init__(
#             metadata=self.dataset.metadata,
#             mask_processors=self.dataset.mask_processor,
#             transforms=self.dataset.transforms,
#             augs_per_img=params['Dataset']['augs_per_img'],
#             valid_size_p=params['Dataset']['val_ratio'],
#             shuffle=self.dataset.shuffle,
#             batch_size=params['Dataset']['batch_size'],
#             workers=params['Dataset']['workers']
#         )

#         self.trainer.__init__(
#             device=self.trainer.device,
#             model=self.trainer.model,
#             dataloaders=self.dataset.create_loaders(),
#             output_path=folder_path,
#             epochs=params['Trainer']['epochs'],
#             checkpoints_ratio=params['Trainer']['checkpoints_ratio']
#             )