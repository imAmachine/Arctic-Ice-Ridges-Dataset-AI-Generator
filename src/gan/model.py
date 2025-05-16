from typing import Dict, List
import torch

from src.gan.models_evaluating import Evaluator, EvalProcessor
from src.gan.custom_evaluators import *
from src.common.interfaces import IGenerativeModel

from src.common.structs import *


@dataclass
class ArchModule:
    model_type: ModelType
    arch: 'torch.nn.Module'
    optimizer: 'torch.optim.Optimizer'
    scheduler: 'torch.optim.lr_scheduler.LRScheduler'
    eval_funcs: Dict[str, Callable]
    eval_settings: Dict


class ModuleTrainManager:
    def __init__(self,
                 device: torch.device,
                 module: 'ArchModule'):
        self.module = module
        self.evaluate_processor = EvalProcessor(device=device, evaluators=self.__build_evaluators())
    
    def __build_evaluators(self) -> List[Evaluator]:
        return [Evaluator(callable_fn=self.module.eval_funcs[k], name=k, **v) for k, v in self.module.eval_settings.items()]
    
    def _process_losses(self, generated_sample: 'torch.Tensor', real_sample: 'torch.Tensor', history_key: ExecPhase) -> 'torch.Tensor':
        self.evaluate_processor.process(generated_sample=generated_sample,
                                        real_sample=real_sample,
                                        exec_phase=history_key)
        
        processed_losses = self.evaluate_processor.evaluators_history[history_key][-1][EvaluatorType.LOSS.value].values()
        
        return sum(processed_losses)
    
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


class GANModel(IGenerativeModel):
    def __init__(self, device: 'torch.device', modules: List[ArchModule], output_path: str, n_critic):
        super().__init__(device, modules, output_path)
        self.n_critic = n_critic
        
    def _training_pipeline(self, input_data, target_data):
        gen, discr = self.t_managers[ModelType.GENERATOR], self.t_managers[ModelType.DISCRIMINATOR]
        
        # тренировка дискриминатора
        for _ in range(self.n_critic):
            with torch.no_grad():
                generated = gen.module.arch(input_data)
            discr.optimization_step(target_data, generated)
        
        # тренировка генератора
        generated = gen.module.arch(input_data)
        gen.optimization_step(target_data, generated)

    def _validation_pipeline(self, input_data, target_data):
        with torch.no_grad():
            generated = self.t_managers[ModelType.GENERATOR].module.arch(input_data)
            for _, t_manager in self.t_managers.items():
                t_manager.valid_step(target_data, generated)

    def _evaluation_pipeline(self, input_data, target_data):
        with torch.no_grad():
            generated = self.t_managers[ModelType.GENERATOR].module.arch(input_data)
            self.save_batch_plot(input_data, target_data, generated)
