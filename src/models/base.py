from dataclasses import field
from abc import ABC, abstractmethod
import os
from tabulate import tabulate

import pandas as pd
from torch import Tensor
from typing import Any, Dict, List, Optional
from src.common.enums import *
from src.models.checkpoint import CheckpointManager
from src.models.gan.evaluators import *


@dataclass
class Evaluator:
    """Датакласс для метрики или лосса и нужной логикой """
    callable_fn: Callable
    name: str
    type: str
    weight: float = field(default=1.0)
    exec_phase: str = ExecPhase.ANY.value
    
    def __call__(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> 'torch.Tensor':   
        return self.callable_fn(generated_sample, real_sample) * self.weight


@dataclass
class Architecture:
    model_type: ModelType
    arch: 'torch.nn.Module'
    optimizer: 'torch.optim.Optimizer'
    scheduler: 'torch.optim.lr_scheduler.LRScheduler'
    evaluators: List[Evaluator]
    
    def __call__(self, inp):
        return self.arch(inp)


class ModuleTrainer:
    """Обёртка над итерацией обучения модели. Подсчёт лоссов и метрик, расчёт градиентов"""
    def __init__(self, device: torch.device, module: 'Architecture'):
        self.module = module
        self.evaluate_processor = EvalProcessor(device=device, evaluators=self.module.evaluators)
    
    def _process_losses(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor', history_key: str) -> 'torch.Tensor':
        self.evaluate_processor.process(
            generated_sample=generated_sample,
            real_sample=real_sample,
            exec_phase=history_key
        )
        
        last_epoch = self.evaluate_processor.evaluators_history[history_key][-1]
        processed_losses = last_epoch[EvaluatorType.LOSS.value].values()
        
        return torch.stack(list(processed_losses)).sum()
    
    def optimization_step(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> float:
        self.module.optimizer.zero_grad()
        loss_tensor = self._process_losses(
            generated_sample, 
            real_sample, 
            history_key=ExecPhase.TRAIN.value
        )
        loss_tensor.backward()
        self.module.optimizer.step()
        return loss_tensor.item()
    
    def valid_step(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor') -> float:
        loss_tensor = self._process_losses(
            generated_sample=generated_sample,
            real_sample=real_sample,
            history_key=ExecPhase.VALID.value
        )
        return loss_tensor.item()


class EvalProcessor:
    """Нужен для подсчёта метрик и лоссов внутри ModuleTrainer"""
    def __init__(self, device: torch.device, evaluators: List[Evaluator]):
        self.device = device
        self.evaluators = evaluators
        self.evaluators_history: Dict[ExecPhase, List] = None
        
        self._init_history_dict()

    def _init_history_dict(self):
        self.evaluators_history = {
            ExecPhase.TRAIN.value: [], 
            ExecPhase.VALID.value: []
        }
    
    def _update_history(self, eval_type: str, phase: str, name: str, value: 'torch.Tensor') -> None:
        current_history = self.evaluators_history[phase][-1]
        current_history[eval_type].update({name: value})

    def process(self, generated_sample, real_sample, exec_phase: str) -> None:
        evaluators_types = [
            member.value
            for member in EvaluatorType
        ]

        self.evaluators_history[exec_phase].append({
            e_type: {} 
            for e_type in evaluators_types
        })
        
        for item in self.evaluators:
            
            if item.exec_phase == ExecPhase.ANY.value or item.exec_phase == exec_phase:
                weighted_tensor = item(generated_sample, real_sample)
                
                self._update_history(
                    eval_type=item.type, 
                    phase=exec_phase, 
                    name=item.name, 
                    value=weighted_tensor
                )
    
    def compute_epoch_summary(self) -> Dict[ExecPhase, Dict[str, Dict[str, float]]]:
        full_result = {}

        for phase, phase_history in self.evaluators_history.items():
            if not phase_history:
                full_result[phase] = {}
                continue

            summary = {}

            for step_result in phase_history:
                for eval_type, evaluators in step_result.items():
                    for name, val in evaluators.items():
                        summary.setdefault(eval_type, {}) \
                        .setdefault(name, []) \
                        .append(val.clone().detach())

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
        self.history: List[Dict[ExecPhase, Dict[ModelType, Dict[str, Dict[str, float]]]]] = []

    def collect(self) -> None:
        """Собирает сводку по эпохе и сбрасывает историю в менеджерах."""
        self.history.append(self._snapshot())
        self._reset()

    def summary_df(self, epoch: int) -> pd.DataFrame:
        """Возвращает детальный DataFrame для заданной эпохи."""
        if not (0 <= epoch < len(self.history)):
            return pd.DataFrame(columns=[
                "Phase", 
                "Model", 
                "Evaluator Type", 
                "Name", 
                "Value"
            ])

        records = [
            {
                "Phase": phase.name,
                "Model": mt.value,
                "Evaluator Type": etype,
                "Name": name,
                "Value": val
            }
            for phase, models in self.history[epoch].items()
            for mt, groups in models.items()
            for etype, items in groups.items()
            for name, val in items.items()
        ]
        
        df = pd.DataFrame(records).fillna("—")
        return df.set_index([
            "Phase", 
            "Model", 
            "Evaluator Type", 
            "Name"
        ]).sort_index()

    def print(self, epoch: Optional[int] = None) -> None:
        """Печатает сводку по эпохе в табличном виде."""
        idx = epoch if epoch is not None else len(self.history) - 1
        df = self.summary_df(idx).reset_index()
        
        for phase, sub in df.groupby("Phase"):
            print(f"[{phase}] ОЦЕНКА, эпоха: {idx + 1}")
            print(
                tabulate(
                    sub[["Model", "Evaluator Type", "Name", "Value"]],
                    headers="keys", tablefmt="fancy_grid", floatfmt=".4f"
                )
            )

    def save_summary(self, output_path: Optional[str] = None, epoch: Optional[int] = None) -> None:        
        # 1) Проверяем, что история не пуста
        if not self.history:
            raise RuntimeError("Нечего сохранять: вызовите collect() до save_summary().")

        idx = len(self.history) - 1 if epoch is None else epoch
        if not (0 <= idx < len(self.history)):
            raise IndexError(f"Epoch {idx} отсутствует в истории (0..{len(self.history)-1}).")

        # 2) Берём все метрики в виде DataFrame
        df = self.summary_df(idx).reset_index()
        if df.empty:
            raise RuntimeError(f"Epoch {idx+1} содержит 0 метрик — нечего сохранять.")

        # 4) Собираем итоговую “горизонтальную” запись
        row: Dict[str, Any] = {}
        for _, r in df.iterrows():
            col = f"{r['Phase']}.{r['Model']}.{r['Evaluator Type']}.{r['Name']}"
            row[col] = r["Value"]

        # 6) Пишем в CSV
        row_df = pd.DataFrame([row])
        write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
        row_df.to_csv(output_path, mode="a", header=write_header, index=False)

    def _snapshot(self) -> Dict:
        """Забирает из менеджеров текущие summary-а по фазам."""
        snapshots = {}
        for phase in (ExecPhase.TRAIN, ExecPhase.VALID):
            phase_snapshot: Dict[ModelType, Dict[str, Dict[str, float]]] = {}
            for mt, mgr in self.managers.items():
                manager_summary = mgr.evaluate_processor.compute_epoch_summary()
                # compute_epoch_summary() keys are phase.value (e.g. "train", "valid")
                phase_snapshot[mt] = manager_summary.get(phase.value, {})
            snapshots[phase] = phase_snapshot
        return snapshots

    def _reset(self) -> None:
        """Сбрасывает историю evaluators в менеджерах."""
        for mgr in self.managers.values():
            mgr.evaluate_processor._init_history_dict()


class GenerativeModel(ABC):
    """Abstract base for generative models: defines training/validation steps."""
    def __init__(self, device: torch.device, checkpoint_map: Dict):
        self.device = device
        self.checkpoint_manager = CheckpointManager(self, checkpoint_map)
        self.evaluators_collector = None
        self.evaluators: Dict[ModelType, List[Evaluator]] = None
        self.trainers: Dict[ModelType, ModuleTrainer] = None
    
    @abstractmethod
    def _train_step(self, inp: Tensor, target: Tensor) -> None:
        pass

    @abstractmethod
    def _valid_step(self, inp: Tensor, target: Tensor) -> None:
        pass
    
    @abstractmethod
    def _init_modules(self, config_section: dict) -> List[Architecture]:
        pass
    
    def _evaluators_from_config(self, config: Dict, device: torch.device):
        from config.registry import LOSSES
        
        for m_type, evals in config.items():
            for eval_name, eval_params in evals.items():
                exec_params = eval_params["execution"]
                init_params = eval_params["init"]
                
                eval_type = EvaluatorType.LOSS.value
                if eval_name in MetricName:
                    eval_type = EvaluatorType.METRIC.value
                
                if exec_params['weight'] > 0.0:
                    cls = LOSSES.get(eval_name)
                    evaluator = Evaluator(
                        callable_fn=cls(**init_params).to(device),
                        name=eval_name,
                        type=eval_type,
                        exec_phase=exec_params["exec_phase"],
                        weight=exec_params["weight"]
                    )
                    
                    self.evaluators[m_type].append(evaluator)
    
    def model_step(self, inp: Tensor, target: Tensor, phase: ExecPhase):
        if phase is ExecPhase.TRAIN:
            self._train_step(inp, target)
        else:
            self._valid_step(inp, target)
    
    def collect_epoch_evaluators(self):
        self.evaluators_collector.collect()
    
    def print_epoch_evaluators(self, epoch_id):
        self.evaluators_collector.print(epoch_id)
    
    def build_train_modules(self, config_section: dict) -> None:
        modules = self._init_modules(config_section)
        
        self.trainers = {
            module.model_type: ModuleTrainer(self.device, module)
            for module in modules
        }
        
        self.evaluators_collector = EvaluatorsCollector(self.trainers)
    
    def evaluators_summary(self, output_path, epoch_id=None):
        self.evaluators_collector.save_summary(output_path, epoch_id)
    
    def checkpoint_save(self, path: str) -> None:
        self.checkpoint_manager.save(path)

    def checkpoint_load(self, path: str) -> None:
        self.checkpoint_manager.load(path)

    def __call__(self, inp: Tensor) -> Tensor:
        if self.trainers is None:
            raise ValueError('Need to init_modules method firts')
        return self.trainers[ModelType.GENERATOR].module(inp)
    