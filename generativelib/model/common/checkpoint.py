import os
import torch

from generativelib.model.common.interfaces import ITorchState


class CheckpointManager:
    @staticmethod
    def save_state(obj: ITorchState, file_path: str) -> None:
        state = obj.to_state_dict()
        torch.save(state, file_path)

    @staticmethod
    def load_state(obj: ITorchState, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл чекпоинта коллекции не найден: {file_path}")
        
        state = torch.load(file_path)
        obj.load_state_dict(state)