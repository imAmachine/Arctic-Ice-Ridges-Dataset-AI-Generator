from __future__ import annotations
import os
from abc import ABC, abstractmethod
from tabulate import tabulate

import torch
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

# Enums
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.enums import ExecPhase

from generativelib.model.evaluators.base import EvalItem, EvalsCollector
from generativelib.model.evaluators.enums import EvaluatorType
from src.visualizer import Visualizer


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
    def __init__(self, arch_module: Arch, evals: List[EvalItem], optimization_params: Dict):
        self.arch_module = arch_module
        self.evals = evals
        self.evals_collector = EvalsCollector()
        
        self.optimizer = torch.optim.Adam(
            self.arch_module.parameters(),
            lr=optimization_params.get('lr', .0005),
            betas=optimization_params.get('betas', (0.9, 0.999))
        )
    
    def loss(self, generated_sample: torch.Tensor, real_sample: torch.Tensor, exec_phase: ExecPhase) -> torch.Tensor:
        loss_tensors = []
        loss_vals_for_history = []

        for item in self.evals:
            if item.exec_phase == ExecPhase.ANY or item.exec_phase == exec_phase:
                val = item(generated_sample, real_sample)
                loss_tensors.append(val.mean())
                loss_vals_for_history.append((item.type, item.name, val.detach().cpu().item()))

        self.evals_collector.collect(loss_vals_for_history, exec_phase)

        return torch.stack(loss_tensors).sum()

    def optimize(self, generated_sample: torch.Tensor, real_sample: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss_tensor = self.loss(
            generated_sample,
            real_sample,
            exec_phase=ExecPhase.TRAIN
        )
        loss_tensor.backward()
        self.optimizer.step()

    def mode_to(self, phase: ExecPhase) -> None:
        if phase == ExecPhase.TRAIN:
            self.arch_module.train()
        
        if phase == ExecPhase.VALID:
            self.arch_module.eval()
    

class ArchOptimizersCollection(list[ArchOptimizer]):
    def by_type(self, model_type: GenerativeModules) -> ArchOptimizer:
        for arch_optimizer in self:
            if arch_optimizer.arch_module.model_type == model_type:
                return arch_optimizer
    
    def add_evals(self, evals: Dict[ModelTypes, List[EvalItem]]) -> None:
        for model_type, evals_list in evals.items():
            cur_optimizer = self.by_type(model_type)
            cur_optimizer.evals.extend(evals_list)
    
    def all_mode_to(self, phase: ExecPhase) -> None:
        for optimizer in self:
            optimizer.mode_to(phase)
    
    def all_clear_history(self) -> None:
        for optimizer in self:
            optimizer.evals_collector.reset_history()
    
    def all_print_phase_summary(self, phase: ExecPhase) -> None:
        # [METHOD AI GENERATED]
        
        headers = ["Type", "Name", "Mean Value"]
        # Получаем имя фазы для вывода и сравнения
        if hasattr(phase, "name"):
            phase_name = phase.name.capitalize()
        else:
            phase_name = str(phase).capitalize()

        for i, optimizer in enumerate(self):
            opt_name = getattr(getattr(optimizer, "arch_module", None), "model_type", None)
            if hasattr(opt_name, "name"):
                opt_name = opt_name.name
            elif opt_name is None:
                opt_name = str(i)
            summary = optimizer.evals_collector.compute_epoch_summary()

            # Собираем строки только по нужной фазе
            rows = []
            for key, mean_val in summary.items():
                exec_phase, typ, name = key
                
                # Универсальное сравнение фаз
                key_phase_name = exec_phase.name.capitalize() if hasattr(exec_phase, "name") else str(exec_phase).capitalize()
                if key_phase_name != phase_name:
                    continue
                mean_str = f"{mean_val:.6f}" if mean_val is not None else "-"
                row = [str(typ.name), str(name), mean_str]
                rows.append(row)

            if rows:
                print(f"\n=== Optimizer: {opt_name} ===\n")
                print(tabulate(rows, headers=headers, tablefmt="github"))
                print()


class BaseHook:
    def __init__(self, interval: int):
        self.interval = interval
    
    def on_phase_begin(self, epoch_id: int, phase: ExecPhase, loader):
        pass
    
    def on_phase_end(self, epoch_id: int, phase: ExecPhase, loader):
        pass


class BaseOptimizationTemplate(ABC):
    def __init__(self, model_params: Dict, arch_optimizers: ArchOptimizersCollection):
        super().__init__()
        self.arch_optimizers = arch_optimizers
        self.model_params = model_params
        
    @abstractmethod
    def _train(self, inp: torch.Tensor, trg: torch.Tensor) -> None:
        pass
    
    @abstractmethod
    def _valid(self, inp: torch.Tensor, trg: torch.Tensor) -> None:
        pass
    
    def step(self, phase: ExecPhase, inp: torch.Tensor, trg: torch.Tensor) -> None:
        if phase == ExecPhase.TRAIN:
            self._train(inp, trg)
        
        if phase == ExecPhase.VALID:
            self._valid(inp, trg)