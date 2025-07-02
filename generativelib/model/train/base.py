import os
import torch
from typing import Dict, List, Optional, Self, Tuple, Type
from abc import ABC, abstractmethod
from tabulate import tabulate
from collections import defaultdict


from generativelib.model.arch.base import ArchModule
from generativelib.model.arch.enums import Modules
from generativelib.model.common.interfaces import ITorchState
from generativelib.model.enums import ExecPhase

from generativelib.model.evaluators.base import LossItem, EvalsCollector
from generativelib.model.evaluators.enums import EvaluatorType


class ModuleOptimizer(ITorchState):
    """Обёртка над ArchModule для обучения. Подсчёт и хранение лоссов и метрик, расчёт градиентов"""
    def __init__(self, arch_module: ArchModule, evals: List[LossItem], optimizer: torch.optim.Optimizer):
        self.module = arch_module
        self.evals = evals
        self.optimizer = optimizer
        self.evals_collector = EvalsCollector()
    
    def to(self, *args, **kwargs) -> Self:
        self.module.to(*args, **kwargs)

        for ev in self.evals:
            ev.to(*args, **kwargs)

        return self
    
    def to_state_dict(self) -> Dict:
        state = {
            "module_state": self.module.to_state_dict(),
            "optim_state": self.optimizer.state_dict()
        }
        return state

    def from_state_dict(self, state_dict: Dict) -> Self:
        self.module.from_state_dict(state_dict["module_state"])
        self.optimizer.load_state_dict(state_dict["optim_state"])
        return self
    
    @classmethod
    def create(cls, arch_module: ArchModule, evals: List[LossItem], optim_info: Dict):
        """Создает объект ModuleOptimizer на основе информации для оптимизатора из optim_info (Dict)"""
        optim_types: Dict[str, Type[torch.optim.Optimizer]] = {
            "adam": torch.optim.Adam,
            "rms": torch.optim.RMSprop
        }
        optim_cls = optim_types.get(optim_info.get('type', ''), torch.optim.Adam)
        optim_params = optim_info.get('params', {})
        
        optimizer = optim_cls(
            arch_module.parameters(),
            **optim_params
        )
        
        return cls(arch_module, evals, optimizer)
    
    def _losses(
        self, 
        generated_sample: torch.Tensor, 
        real_sample: torch.Tensor,
        mask: torch.Tensor,
        exec_phase: ExecPhase
    ) -> Tuple[torch.Tensor, List[Tuple[EvaluatorType, str, float]]]:
        loss_tensor = torch.tensor(0.0, device=generated_sample.device, dtype=generated_sample.dtype)
        losses_vals: List[Tuple[EvaluatorType, str, float]] = []

        # подсчёт лоссов
        for item in self.evals:
            if item.exec_phase == ExecPhase.ANY or item.exec_phase == exec_phase:
                val = item(generated_sample, real_sample)
                val_mean = val.mean()
                
                loss_tensor = loss_tensor + val.mean()
                losses_vals.append((
                    EvaluatorType.LOSS, 
                    item.name, 
                    val_mean.detach().cpu().item()
                ))
        
        # добавление общего loss
        losses_vals.append((
            EvaluatorType.LOSS,
            "total",
            loss_tensor.mean().item()
        ))
        
        return loss_tensor, losses_vals

    def optimize(self, generated: torch.Tensor, real: torch.Tensor, mask: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        total_loss, batch_evals = self._losses(generated, real, mask, ExecPhase.TRAIN)

        self.evals_collector.collect(batch_evals, ExecPhase.TRAIN)

        total_loss.backward()
        self.optimizer.step()

    def validate(self, generated: torch.Tensor, real: torch.Tensor, mask: torch.Tensor) -> None:
        with torch.no_grad():
            _, batch_evals = self._losses(generated, real, mask, ExecPhase.VALID)
        self.evals_collector.collect(batch_evals, ExecPhase.VALID)

    def mode_to(self, phase: ExecPhase) -> Self:
        if phase == ExecPhase.TRAIN:
            self.module.train()
        
        if phase == ExecPhase.VALID:
            self.module.eval()
        
        self.evals_collector.reset_history()
        
        return self


class ModuleOptimizersCollection(list[ModuleOptimizer], ITorchState):
    """Обёртка для управления коллекцией ModuleOptimizer"""
    def to(self, *args, **kwargs) -> Self:
        for optim in self:
            optim.to(*args, **kwargs)
        return self
    
    def to_state_dict(self) -> Dict:
        return {optim.module.model_type.name.lower(): optim.to_state_dict() for optim in self}
    
    def from_state_dict(self, state_dict: Dict) -> Self:
        for optim in self:
            optim.from_state_dict(state_dict[optim.module.model_type.name.lower()])
        return self
    
    def by_type(self, model_type: Modules) -> ModuleOptimizer:
        results = list(set(arch_optimizer for arch_optimizer in self if arch_optimizer.module.model_type == model_type))
        if len(results) > 0:
            return results[0]
        
        raise ValueError(f'No optimizers found by type: {model_type.name}')
    
    def add_losses(self, evals: Dict[Modules, List[LossItem]]) -> Self:
        for model_type, evals_list in evals.items():
            cur_optimizer = self.by_type(model_type)
            cur_optimizer.evals.extend(evals_list)
        return self
    
    def mode_to(self, phase: ExecPhase) -> Self:
        for optimizer in self:
            optimizer.mode_to(phase)
        return self


class OptimizationTemplate(ABC):
    def __init__(self, model_params: Dict, module_optimizers: ModuleOptimizersCollection):
        super().__init__()
        self.optimizers = module_optimizers
        self.params = model_params
    
    @abstractmethod
    def _train(self, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:
        pass
    
    @abstractmethod
    def _valid(self, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:
        pass
    
    @abstractmethod
    def generate(self, inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def mode_to(self, phase: ExecPhase) -> Self:
        self.optimizers.mode_to(phase)
        return self
    
    def to(self, device: torch.device) -> Self:
        self.optimizers.to(device)
        return self
    
    def step(self, phase: ExecPhase, inp: torch.Tensor, trg: torch.Tensor, mask: torch.Tensor) -> None:
        if phase == ExecPhase.TRAIN:
            self._train(inp, trg, mask)
        
        if phase == ExecPhase.VALID:
            self._valid(inp, trg, mask)
    
    def _print_phase_summary(self, phase_name: str, summary: Dict) -> List[str]:
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

        for optimizer in self.optimizers:
            opt_name = optimizer.module.model_type.name
            summary = optimizer.evals_collector.compute_epoch_summary()
            rows = self._print_phase_summary(phase_name, summary)
            
            if rows:
                print(f"\n=== Optimizer: {opt_name} ===\n")
                print(tabulate(rows, headers=headers, tablefmt="github"))
                print()