import torch

from generativelib.model.train.base import ModuleOptimizersCollection
    
# ================
# НУЖНО ПЕРЕРАБОТАТЬ В ХУК ДЛЯ ЧЕКПОИНТА [WIP]
# ================
class Checkpoint:
    """Сlass for saving and loading a model."""
    def __init__(self, optimizers: ModuleOptimizersCollection):
        self.optimizers = optimizers

    def _traverse_path(self, path: tuple):
        """Рекурсивно проходит по пути из кортежа"""
        obj = self.optimizers
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

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.optimizers.device, weights_only=False)
        for role, components in self.checkpoint_map.items():
            for name, attr_path in components.items():
                obj = self._traverse_path(attr_path)
                obj.load_state_dict(checkpoint[role][name])
        print(f"Checkpoint loaded from {path}")