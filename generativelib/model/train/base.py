from __future__ import annotations
import os
from abc import ABC, abstractmethod
from tabulate import tabulate

import torch
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional

# Enums
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.enums import ExecPhase

from generativelib.model.evaluators.base import EvalItem, EvaluateProcessor
from generativelib.model.evaluators.enums import EvaluatorType


class Arch(torch.nn.Module):
    def __init__(
        self,
        model_type: GenerativeModules,
        module: torch.nn.Module
    ):
        super().__init__()
        self.model_type = model_type
        self.module = module

    def forward(self, x):
        return self.module(x)
    

class ArchOptimizer:
    """Обёртка над итерацией обучения модели. Подсчёт лоссов и метрик, расчёт градиентов"""
    def __init__(self, arch_module: Arch, eval_processor: EvaluateProcessor, optimization_params: Dict):
        self.arch_module = arch_module
        self.evaluate_processor = eval_processor
        self.optimizer = torch.optim.Adam(
            self.arch_module.parameters(),
            lr=optimization_params['lr'],
            betas=optimization_params.get('betas', (0.9, 0.999))
        )
    
    def loss(self, generated_sample: torch.Tensor, real_sample: torch.Tensor, exec_phase: ExecPhase) -> torch.Tensor:
        self.evaluate_processor.process(
            generated_sample=generated_sample,
            real_sample=real_sample,
            exec_phase=exec_phase
        )

        last_epoch: Dict[str, Dict[str, torch.Tensor]] = self.evaluate_processor.evals_history[exec_phase][-1]
        processed = last_epoch[EvaluatorType.LOSS].values()
        return torch.stack([t.mean() for t in processed]).sum()

    def optimize(self, generated_sample: torch.Tensor, real_sample: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        loss_tensor = self.loss(
            generated_sample,
            real_sample,
            exec_phase=ExecPhase.TRAIN
        )

        loss_tensor.backward()
        self.optimizer.step()

        return loss_tensor.item()

    def mode_to(self, phase: ExecPhase):
        if phase == ExecPhase.TRAIN:
            self.arch_module.train()
        
        if phase == ExecPhase.VALID:
            self.arch_module.eval()
    

class ArchOptimizersCollection(list[ArchOptimizer]):
    def by_type(self, model_type: GenerativeModules) -> ArchOptimizer:
        for arch_optimizer in self:
            if arch_optimizer.arch_module.model_type == model_type:
                return arch_optimizer
    
    def add_evals(self, evals: Dict[ModelTypes, List[EvalItem]]):
        for model_type, evals_list in evals.items():
            cur_optimizer = self.by_type(model_type)
            cur_optimizer.evaluate_processor.evals.extend(evals_list)
    
    def all_mode_to(self, phase: ExecPhase):
        for optimizer in self:
            if phase == ExecPhase.TRAIN:
                optimizer.arch_module.train()
            
            if phase == ExecPhase.VALID:
                optimizer.arch_module.eval()


class EvaluatorsCollector:
    def __init__(self, modules_optimizers: Dict[ModelTypes, ArchOptimizer]):
        self.managers = modules_optimizers
        self.history: List[Dict[ExecPhase, Dict[ModelTypes, Dict[str, Dict[str, float]]]]] = []

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
            phase_snapshot: Dict[ModelTypes, Dict[str, Dict[str, float]]] = {}
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


class BaseTrainTemplate(ABC):
    def __init__(self, model_params: Dict, arch_optimizers: ArchOptimizersCollection):
        super().__init__()
        self.arch_optimizers = arch_optimizers
        self.model_params = model_params
    
    @abstractmethod
    def _train(self, inp: torch.Tensor, trg: torch.Tensor):
        pass
    
    @abstractmethod
    def _valid(self, inp: torch.Tensor, trg: torch.Tensor):
        pass
    
    def step(self, phase: ExecPhase, inp: torch.Tensor, trg: torch.Tensor):
        self.arch_optimizers.all_mode_to(phase)
        if phase == ExecPhase.TRAIN:
            self._train(inp, trg)
        
        if phase == ExecPhase.VALID:
            self._valid(inp, trg)
