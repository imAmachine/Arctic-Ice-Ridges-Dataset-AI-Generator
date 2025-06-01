from typing import Dict, List, Tuple

import torch
from generativelib.config_tools.base import ConfigModelSerializer
from generativelib.config_tools.default_values import ENABLED_KEY, EVALS_KEY, EXEC_PHASE_KEY, EXECUTION_KEY, INIT_KEY, MASK_PROCESSORS_KEY, MODEL_PARAMS_KEY, MODELS_KEY, MODULES_KEY, OPTIMIZATION_PARAMS_KEY, PARAMS_KEY, WEIGHT_KEY
from generativelib.dataset.base import BaseMaskProcessor
from generativelib.dataset.mask_processors import MASK_PROCESSORS
from generativelib.model.arch.enums import ModelTypes, GenerativeModules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.base import LOSSES, EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, MetricName
from generativelib.model.train.base import Arch, ArchOptimizer, ArchOptimizersCollection


class TrainConfigSerializer(ConfigModelSerializer):
    def __init__(self, config_folder_path: str, phase: ExecPhase):
        super().__init__(config_folder_path, phase)
        self._mask_processors: Dict = self.config[MASK_PROCESSORS_KEY]
        self._models: Dict = self.config[MODELS_KEY]
    
    def serialize_mask_processors(self) -> List[BaseMaskProcessor]:
        processors: List[BaseMaskProcessor] = []
        
        for name, values in self._mask_processors.items():
            if values[ENABLED_KEY]:
                cls = MASK_PROCESSORS.get(name)
                processors.append(cls(**values[PARAMS_KEY]))
        
        return processors
    
    def serialize_evaluators(self, device: torch.device, eval_info: Dict) -> List[EvalItem]:
        evals = []
        
        for eval_name, eval_params in eval_info.items():
            exec_params = eval_params[EXECUTION_KEY]
            init_params = eval_params[INIT_KEY]
            
            eval_type = EvaluatorType.LOSS
            if eval_name in MetricName:
                eval_type = EvaluatorType.METRIC
            
            if exec_params[WEIGHT_KEY] > 0.0:
                cls = LOSSES.get(eval_name)
                exec_phase = exec_params[EXEC_PHASE_KEY]
                
                evaluator = EvalItem(
                    callable_fn=cls(**init_params).to(device),
                    name=eval_name,
                    type=eval_type,
                    exec_phase=ExecPhase[exec_phase],
                    weight=exec_params[WEIGHT_KEY]
                )
                
                evals.append(evaluator)
        return evals
    
    def serialize_model(self, device: torch.device, model_type: ModelTypes) -> Tuple[ArchOptimizersCollection, Dict]:
        model_dict: Dict = self._models.get(model_type.name)
        optimization_params: Dict = model_dict.get(OPTIMIZATION_PARAMS_KEY)
        
        model_params = model_dict.get(MODEL_PARAMS_KEY)
        modules: Dict = model_dict.get(MODULES_KEY)
        
        arch_params = self.params_by_section(section='arch', keys=['in_ch', 'f_base'])
        
        arch_collect = ArchOptimizersCollection()
        for m_name, module_info in modules.items():
            cls = GenerativeModules[m_name].value
            arch = Arch(GenerativeModules[m_name], cls(**arch_params).to(device))
            evals = self.serialize_evaluators(device, module_info.get(EVALS_KEY))
            arch_collect.append(
                ArchOptimizer(arch_module=arch, evals=evals, optimization_params=optimization_params)
            )
            
        return arch_collect, model_params