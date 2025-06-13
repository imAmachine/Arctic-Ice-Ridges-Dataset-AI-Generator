import os
import torch

from generativelib.model.common.interfaces import ITorchState


class CheckpointManager:
    @staticmethod
    def save_state(obj: ITorchState, folder_path: str) -> None:
        state = obj.to_state_dict()
        torch.save(state, folder_path)

    @staticmethod
    def load_state(obj: ITorchState, folder_path: str) -> None:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Файл чекпоинта коллекции не найден: {folder_path}")
        
        state = torch.load(folder_path, weights_only=False)
        obj.from_state_dict(state)