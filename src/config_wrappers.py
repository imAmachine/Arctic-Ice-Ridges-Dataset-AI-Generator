from typing import Dict, List, Tuple

import torch
from generativelib.config_tools.base import ConfigModelSerializer
from generativelib.config_tools.default_values import ARCH_PARAMS_KEY, ENABLED_KEY, EVALS_KEY, EXEC_PHASE_KEY, EXECUTION_KEY, INIT_KEY, MASK_PROCESSORS_KEY, MODEL_PARAMS_KEY, MODELS_KEY, MODULES_KEY, OPTIMIZER_KEY, PARAMS_KEY, WEIGHT_KEY
from generativelib.dataset.base import BaseMaskProcessor
from generativelib.model.arch.enums import ModelTypes, GenerativeModules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.train.base import ArchModule, ModuleOptimizer, ArchOptimizersCollection


class TrainConfigSerializer(ConfigModelSerializer):
    def __init__(self, config_folder_path: str, phase: ExecPhase):
        super().__init__(config_folder_path, phase)
        self._mask_processors: Dict = self.config[MASK_PROCESSORS_KEY]
        self._models: Dict = self.config[MODELS_KEY]
    
    def serialize_all_masks(self) -> List[BaseMaskProcessor]:
        processors: List[BaseMaskProcessor] = []
        
        for name, values in self._mask_processors.items():
            proc = BaseMaskProcessor.from_dict(name, values)
            
            if proc: processors.append(proc)
        
        return processors
    
    def serialize_all_eval_items(self, device: torch.device, eval_info: Dict) -> List[EvalItem]:
        evals: List[EvalItem] = []
        
        for eval_name, eval_params in eval_info.items():
            eval_it = EvalItem.from_dict(device, eval_name, eval_params)
            
            if eval_it: evals.append(eval_it)
            
        return evals
    
    def serialize_module_optimizer(self, device: torch.device, module_name: str, arch_info: Dict, evals_info: Dict, optim_info: Dict) -> ModuleOptimizer:
        arch_module = ArchModule.from_dict(device, module_name, arch_info)
        evals = self.serialize_all_eval_items(device, evals_info)
        optimizer = ModuleOptimizer.from_dict(arch_module, optim_info)
        
        return ModuleOptimizer(
            arch_module=arch_module, 
            evals=evals,
            optimizer=optimizer
        )
    
    def serialize_optimize_collection(self, device: torch.device, model_type: ModelTypes) -> ArchOptimizersCollection:
        arch_collect = ArchOptimizersCollection()
        
        modules = self._models.get(model_type.name.lower()).get(MODULES_KEY)
        
        for module_name, module_info in modules.items():
            arch_info = module_info.get(ARCH_PARAMS_KEY, {})
            evals_info = module_info.get(EVALS_KEY, {})
            optim_info = module_info.get(OPTIMIZER_KEY, {})
            
            module_optimizer = self.serialize_module_optimizer(device, module_name, arch_info, evals_info, optim_info)
            arch_collect.append(module_optimizer)
        
        return arch_collect
    
    def get_model_params(self, model_type: GenerativeModules):
        model_dict: Dict = self._models.get(model_type.name.lower())
        return model_dict.get(MODEL_PARAMS_KEY)