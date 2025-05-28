import torch
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.base import GenerativeModel

class CheckpointManager:
    """Сlass for saving and loading a model."""
    def __init__(self, model: 'GenerativeModel', checkpoint_map: dict):
        self.model = model
        self.checkpoint_map = checkpoint_map

    def _traverse_path(self, path: tuple):
        """Рекурсивно проходит по пути из кортежа"""
        obj = self.model
        for item in path:
            if isinstance(obj, dict):
                obj = obj.get(item)
            else:
                obj = getattr(obj, item, None)
            if obj is None:
                raise ValueError(f"Invalid checkpoint path: {path}")
        return obj

    def save(self, path: str):
        checkpoint = {}
        for role, components in self.checkpoint_map.items():
            checkpoint[role] = {}
            for name, attr_path in components.items():
                obj = self._traverse_path(attr_path)
                checkpoint[role][name] = obj.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path: str, debug: bool = True):
        try:
            checkpoint = torch.load(path, map_location=self.model.device, weights_only=False)

            def try_load(use_copy: bool):
                for role, components in self.checkpoint_map.items():
                    for name, attr_path in components.items():
                        obj = self._traverse_path(attr_path)
                        target = copy.deepcopy(obj) if use_copy else obj
                        target.load_state_dict(checkpoint[role][name])

            if debug:
                try_load(use_copy=True)
            try_load(use_copy=False) 
            print(f"Checkpoint loaded from {path}")
        except Exception as e:
            print(f"Checkpoint load failed: {e}")