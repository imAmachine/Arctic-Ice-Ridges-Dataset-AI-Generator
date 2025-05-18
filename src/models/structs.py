from collections import defaultdict
from functools import wraps
import os

from torch import Tensor
from src.models.evaluating import EvalProcessor, Evaluator
from src.models.gan.gan_evaluators import *
from src.common.enums import *
from typing import Dict, List, Optional
from matplotlib import pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod
from tabulate import tabulate

@dataclass
class ArchModule:
    model_type: ModelType
    arch: 'torch.nn.Module'
    optimizer: 'torch.optim.Optimizer'
    scheduler: 'torch.optim.lr_scheduler.LRScheduler'
    eval_funcs: Dict[str, Callable]
    eval_settings: Dict


class ModuleTrainer:
    """Обёртка над итерацией обучения модели. Подсчёт лоссов и метрик, расчёт градиентов"""
    def __init__(self,device: torch.device, module: 'ArchModule'):
        self.module = module
        self.evaluate_processor = EvalProcessor(device=device, evaluators=self.__build_evaluators())
    
    def __build_evaluators(self) -> List[Evaluator]:
        return [Evaluator(callable_fn=self.module.eval_funcs[k], name=k, **v) for k, v in self.module.eval_settings.items()]
    
    def _process_losses(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor', history_key: ExecPhase) -> 'torch.Tensor':
        self.evaluate_processor.process(generated_sample=generated_sample,
                                        real_sample=real_sample,
                                        exec_phase=history_key)
        processed_losses = self.evaluate_processor.evaluators_history[history_key][-1][EvaluatorType.LOSS.value].values()
        
        return torch.stack(list(processed_losses)).sum()
    
    def optimization_step(self, real_sample: 'torch.Tensor', generated_sample: 'torch.Tensor') -> float:
        self.module.optimizer.zero_grad()
        loss_tensor = self._process_losses(generated_sample, real_sample, history_key=ExecPhase.TRAIN)
        loss_tensor.backward()
        self.module.optimizer.step()
        return loss_tensor.item()
    
    def valid_step(self, real_sample: 'torch.Tensor', generated_sample: 'torch.Tensor') -> float:
        loss_tensor = self._process_losses(generated_sample=generated_sample,
                                          real_sample=real_sample,
                                          history_key=ExecPhase.VALID)
        return loss_tensor.item()


class EvaluatorsCollector:
    def __init__(self, managers: Dict[ModelType, ModuleTrainer]):
        self.managers = managers
        self.history_epochs: List[Dict[ExecPhase, Dict[ModelType, Dict[str, Dict[str, float]]]]] = []

    def collect_epoch_summary(self) -> None:
        summary = self.__summary_from_managers()
        self.history_epochs.append(summary)
        self.__reset_managers()

    def summary_df(self,epoch_id: int) -> pd.DataFrame:
        if epoch_id < 0 or epoch_id >= len(self.history_epochs):
            return pd.DataFrame(columns=["Phase", "Model", "Evaluator Type", "Name", "Value"])

        epoch_summary = self.history_epochs[epoch_id]
        rows = []
        for ph, models_evaluators in epoch_summary.items():
            for model_type, eval_groups in models_evaluators.items():
                for eval_type, evaluators in eval_groups.items():
                    for name, value in evaluators.items():
                        rows.append({
                            "Phase": ph.name,
                            "Model": model_type.value,
                            "Evaluator Type": eval_type,
                            "Name": name,
                            "Value": value
                        })

        df = pd.DataFrame(rows)
        df = df.fillna("—")
        return df.set_index(["Phase", "Model", "Evaluator Type", "Name"]).sort_index()

    def print_summary(self, epoch_id: Optional[int] = None) -> None:
        if epoch_id is None:
            epoch_id = len(self.history_epochs) - 1

        df = self.summary_df(epoch_id).reset_index()

        for phase_name in df["Phase"].unique():
            phase_df = df[df["Phase"] == phase_name]
            print(f"\Evaluators for phase: {phase_name}, epoch: {epoch_id + 1}")
            display_df = phase_df[["Model", "Evaluator Type", "Name", "Value"]]
            print(tabulate(display_df, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

    def reset_history(self) -> None:
        self.history_epochs.clear()

    def __summary_from_managers(self) -> Dict[ExecPhase, Dict[ModelType, Dict[str, Dict[str, float]]]]:
        result: Dict[ExecPhase, Dict[ModelType, Dict[str, Dict[str, float]]]] = {}
        for phase in [ExecPhase.TRAIN, ExecPhase.VALID]:
            phase_summary: Dict[ModelType, Dict[str, Dict[str, float]]] = {}
            for model_type, mgr in self.managers.items():
                full_summary = mgr.evaluate_processor.compute_epoch_summary()
                
                evaluators_for_phase = full_summary.get(phase, {})
                phase_summary[model_type] = evaluators_for_phase
            result[phase] = phase_summary
        return result
    
    def __reset_managers(self) -> None:
        for mgr in self.managers.values():
            mgr.evaluate_processor.reset_history()



class Visualizer:
    """Handles saving grids of input, generated, and target samples."""
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def save(self, inp: Tensor, target: Tensor, gen: Tensor, phase: ExecPhase, samples: int = 3) -> None:
        cols = min(samples, inp.size(0), 5)
        plt.figure(figsize=(12, 12), dpi=300)
        for row_idx, batch in enumerate((inp, gen, target)):
            for col_idx in range(cols):
                img = batch[col_idx].cpu().squeeze()
                ax = plt.subplot(3, cols, row_idx * cols + col_idx + 1)
                ax.imshow(img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"{['Input', 'Gen', 'Target'][row_idx]} {col_idx+1}")
                ax.axis('off')
        plt.suptitle(f"Phase: {phase.value}", y=1.02)
        plt.tight_layout(pad=3)
        path = os.path.join(self.output_path, f"{phase.value}.png")
        plt.savefig(path)
        plt.close()


class CheckpointManager:
    """Сlass for saving and loading a model."""
    def __init__(self, model: 'BaseModel', checkpoint_map: dict):
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

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.model.device, weights_only=False)
        for role, components in self.checkpoint_map.items():
            for name, attr_path in components.items():
                obj = self._traverse_path(attr_path)
                obj.load_state_dict(checkpoint[role][name])
        print(f"Checkpoint loaded from {path}")


class BaseModel(ABC):
    """Abstract base for generative models: defines training/validation steps."""
    def __init__(self, device: torch.device, checkpoint_map: Dict):
        self.device = device
        self.checkpoint_manager = CheckpointManager(self, checkpoint_map)
        self.evaluators = None
        self.trainers = None
    
    @abstractmethod
    def _train_step(self, inp: Tensor, target: Tensor) -> None:
        pass

    @abstractmethod
    def _valid_step(self, inp: Tensor, target: Tensor) -> None:
        pass
    
    @abstractmethod
    def _init_modules(self, config_section: dict) -> List[ArchModule]:
        pass
    
    def model_step(self, inp: Tensor, target: Tensor, phase: ExecPhase):
        if phase is ExecPhase.TRAIN:
            self._train_step(inp, target)
        else:
            self._valid_step(inp, target)
    
    def collect_epoch_evaluators(self):
        self.evaluators.collect_epoch_summary()
    
    def print_evaluators(self, epoch_id):
        self.evaluators.print_summary(epoch_id)
    
    def build(self, config_section: dict) -> None:
        modules = self._init_modules(config_section)
        
        self.trainers = {
            module.model_type: ModuleTrainer(self.device, module)
            for module in modules
        }
        
        self.evaluators = EvaluatorsCollector(self.trainers)
    
    def save(self, path: str) -> None:
        self.checkpoint_manager.save(path)

    def load(self, path: str) -> None:
        self.checkpoint_manager.load(path)

    def __call__(self, inp: Tensor) -> Tensor:
        if self.trainers is None:
            raise ValueError('Need to init_modules method firts')
        return self.trainers[ModelType.GENERATOR].module.arch(inp)
    