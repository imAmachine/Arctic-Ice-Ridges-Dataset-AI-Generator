from __future__ import annotations
import os
from abc import ABC, abstractmethod
from tabulate import tabulate

import torch
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

# Enums
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.enums import ExecPhase

from generativelib.model.evaluators.base import EvalItem, EvalsCollector
from generativelib.model.evaluators.enums import EvaluatorType
from generativelib.common.visualizer import Visualizer


class ArchModule(torch.nn.Module):
    def __init__(
        self,
        model_type: GenerativeModules,
        module: torch.nn.Module
    ):
        super().__init__()
        self.model_type = model_type
        self.module = module

    @classmethod
    def from_dict(cls, device: torch.device, module_name: str, arch_params: Dict):
        module_cls = GenerativeModules[module_name.upper()].value
        arch_module = module_cls(**arch_params).to(device)
        
        return cls(GenerativeModules[module_name.upper()], arch_module)
    
    def forward(self, x):
        return self.module(x)
    

class ModuleOptimizer:
    """Обёртка над ArchModule для обучения. Подсчёт лоссов и метрик, расчёт градиентов"""
    def __init__(self, arch_module: ArchModule, evals: List[EvalItem], optimizer: torch.optim.Optimizer):
        self.arch = arch_module
        self.evals = evals
        self.optimizer = optimizer
        self.evals_collector = EvalsCollector()
        
    @classmethod
    def create(cls, arch_module: ArchModule, evals: List[EvalItem], optim_info: Dict):
        """Создает объект ModuleOptimizer на основе информации для оптимизатора из optim_info (Dict)"""
        optim_types: Dict[str, torch.optim.Optimizer] = {
            "adam": torch.optim.Adam,
            "rms": torch.optim.RMSprop
        }
        
        optim_cls = optim_types.get(optim_info.get('type'), torch.optim.Adam)
        optim_params = optim_info.get('params')
        
        optimizer = optim_cls(
            arch_module.parameters(),
            **optim_params
        )
        
        return cls(arch_module, evals, optimizer)
    
    def _losses(self, generated_sample: torch.Tensor, real_sample: torch.Tensor, exec_phase: ExecPhase) -> Tuple[torch.Tensor, List[Tuple[str, str, float]]]:
        loss_tensor = torch.tensor(0.0, device=generated_sample.device, dtype=generated_sample.dtype)
        losses_vals: List[Tuple[str, str, float]] = []

        # подсчёт лоссов
        for item in self.evals:
            if item.exec_phase == ExecPhase.ANY or item.exec_phase == exec_phase:
                val = item(generated_sample, real_sample)
                loss_tensor = loss_tensor + val.mean()
                losses_vals.append((
                    item.type, 
                    item.name, 
                    val.detach().cpu().item()
                ))
        
        # добавление общего loss
        losses_vals.append((
            EvaluatorType.LOSS,
            "total",
            loss_tensor.mean().item()
        ))
        
        return loss_tensor, losses_vals

    def optimize(self, generated: torch.Tensor, real: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        total_loss, batch_evals = self._losses(generated, real, ExecPhase.TRAIN)

        self.evals_collector.collect(batch_evals, ExecPhase.TRAIN)

        total_loss.backward()
        self.optimizer.step()

    def validate(self, generated: torch.Tensor, real: torch.Tensor) -> None:
        with torch.no_grad():
            _, batch_evals = self._losses(generated, real, ExecPhase.VALID)
        self.evals_collector.collect(batch_evals, ExecPhase.VALID)

    def mode_to(self, phase: ExecPhase) -> None:
        if phase == ExecPhase.TRAIN:
            self.arch.train()
        
        if phase == ExecPhase.VALID:
            self.arch.eval()
        
        self.evals_collector.reset_history()


class ArchOptimizersCollection(list[ModuleOptimizer]):
    def by_type(self, model_type: GenerativeModules) -> ModuleOptimizer:
        for arch_optimizer in self:
            if arch_optimizer.arch.model_type == model_type:
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
    
    def _print_phase_summary(self, phase_name: str, headers: List[str], opt_name: str, summary: Dict) -> List[str]:
        # Cтроки только по нужной фазе
        rows = []
        for key, mean_val in summary.items():
            exec_phase, typ, name = key
            
            # Универсальное сравнение фаз
            key_phase_name = exec_phase.name.capitalize()
            if key_phase_name != phase_name:
                continue
            mean_str = f"{mean_val:.6f}" if mean_val is not None else "-"
            row = [str(typ.name), str(name), mean_str]
            rows.append(row)
        return rows
    
    def all_print_phase_summary(self, phase: ExecPhase) -> None:
        # [METHOD AI GENERATED]
        
        headers = ["Type", "Name", "Mean Value"]
        # Получаем имя фазы для вывода и сравнения
        phase_name = phase.name.capitalize()

        for optimizer in self:
            opt_name = optimizer.arch.model_type.name
            summary = optimizer.evals_collector.compute_epoch_summary()
            rows = self._print_phase_summary(phase_name, headers, opt_name, summary)
            
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