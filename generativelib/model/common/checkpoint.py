import os
import torch

from generativelib.model.common.interfaces import ITorchState


class CheckpointManager:
    @staticmethod
    def save_state(obj: ITorchState, folder_path: str) -> None:
        state = obj.to_state_dict()
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, 'checkpoint.pt')
        torch.save(state, file_path)

    @staticmethod
    def load_state(obj: ITorchState, folder_path: str) -> None:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Папка чекпоинта коллекции не найдена: {folder_path}")
        
        file_path = os.path.join(folder_path, 'checkpoint.pt')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл чекпоинта коллекции не найден: {file_path}")
        
        state = torch.load(file_path, weights_only=False)
        obj.from_state_dict(state)