import os
import pandas as pd
from typing import Dict

from src.common.enums import ExecPhase
from src.models.train import Trainer

class TestResultSaver:
    def __init__(self, result_path: str):
        self.result_path = result_path

    def save(self, folder_path: str, params: Dict, trainer: Trainer):
        summary_row = {'folder': folder_path, **params}
        last_metrics = trainer.model.evaluators.history_epochs[-1]
        val_metrics = last_metrics.get(ExecPhase.VALID, {})
        
        for model_type, eval_groups in val_metrics.items():
            metrics = eval_groups.get('Metric', {})
            for metric_name, value in metrics.items():
                summary_row[metric_name] = value

        df = pd.DataFrame([summary_row])
        df.to_csv(self.result_path, mode='a' if os.path.exists(self.result_path) else 'w',
                  header=not os.path.exists(self.result_path), index=False)