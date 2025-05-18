from dataclasses import field
from abc import ABC, abstractmethod
from tabulate import tabulate

import pandas as pd
from torch import Tensor
from typing import Dict, List, Optional

from src.common.enums import *
from src.models.checkpoint import CheckpointManager
from src.models.gan.evaluators import *

@dataclass
class Architecture:
    model_type: ModelType
    arch: 'torch.nn.Module'
    optimizer: 'torch.optim.Optimizer'
    scheduler: 'torch.optim.lr_scheduler.LRScheduler'
    eval_funcs: Dict[str, Callable]
    eval_settings: Dict


@dataclass
class Evaluator:
    """Датакласс для метрики или лосса и нужной логикой """
    callable_fn: Callable
    name: str
    type: str
    weight: float = field(default=1.0)
    exec_phase: str = ExecPhase.ANY.value
    
    def __call__(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> 'torch.Tensor':   
        if self.weight <= 0.0:
            return torch.tensor(0.0, device=generated_sample.device)
        return self.callable_fn(generated_sample, real_sample) * self.weight


class ModuleTrainer:
    """Обёртка над итерацией обучения модели. Подсчёт лоссов и метрик, расчёт градиентов"""
    def __init__(self,device: torch.device, module: 'Architecture'):
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


class EvalProcessor:
    """Нужен для подсчёта метрик и лоссов внутри ModuleTrainer"""
    def __init__(self, device: torch.device, evaluators: List[Evaluator] = []):
        self.device = device
        self.evaluators = evaluators
        self.evaluators_history: Dict[ExecPhase, List] = {
            ExecPhase.TRAIN: [], ExecPhase.VALID: []
        }

    def _update_history(self, eval_type: str, phase: ExecPhase, name: str, value: 'torch.Tensor') -> None:
        current_history = self.evaluators_history[phase][-1]
        current_history[eval_type].update({name: value})

    def process(self, generated_sample, real_sample, exec_phase: str) -> None:
        evaluators_types = [member.value for member in EvaluatorType]
        self.evaluators_history[exec_phase].append({e_type: {} for e_type in evaluators_types})
        
        for item in self.evaluators:
            phase_value = exec_phase.value
            if item.exec_phase == ExecPhase.ANY.value or item.exec_phase == phase_value:
                weighted_tensor = item(generated_sample, real_sample)
                self._update_history(item.type, exec_phase, item.name, weighted_tensor)

    def reset_history(self):
        self.evaluators_history = {ExecPhase.TRAIN: [], ExecPhase.VALID: []}
    
    def compute_epoch_summary(self) -> Dict[ExecPhase, Dict[str, Dict[str, float]]]:
        full_result = {}

        for phase, phase_history in self.evaluators_history.items():
            if not phase_history:
                full_result[phase] = {}
                continue

            summary = {}

            for step_result in phase_history:
                for eval_type, metrics in step_result.items():
                    for name, val in metrics.items():
                        summary.setdefault(eval_type, {}).setdefault(name, []).append(val.clone().detach())

            averaged = {
                eval_type: {
                    name: torch.stack(values).mean().item()
                    for name, values in metrics.items()
                }
                for eval_type, metrics in summary.items()
            }

            full_result[phase] = averaged

        return full_result


class EvaluatorsCollector:
    def __init__(self, managers: Dict[ModelType, ModuleTrainer]):
        self.managers = managers
        self.history_epochs: List[Dict[ExecPhase, Dict[ModelType, Dict[str, Dict[str, float]]]]] = []

    def collect_epoch_summary(self) -> None:
        summary = self.__summary_from_managers()
        self.history_epochs.append(summary)
        self.__reset_managers()

    def summary_df(self, epoch_id: int) -> pd.DataFrame:
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
            print(f"\[{phase_name}] ОЦЕНКА, эпоха: {epoch_id + 1}")
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


class GenerativeModel(ABC):
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
    def _init_modules(self, config_section: dict) -> List[Architecture]:
        pass
    
    def model_step(self, inp: Tensor, target: Tensor, phase: ExecPhase):
        if phase is ExecPhase.TRAIN:
            self._train_step(inp, target)
        else:
            self._valid_step(inp, target)
    
    def collect_epoch_evaluators(self):
        self.evaluators.collect_epoch_summary()
    
    def print_epoch_evaluators(self, epoch_id):
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
    