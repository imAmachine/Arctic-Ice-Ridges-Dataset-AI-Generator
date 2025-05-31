from typing import Dict, List

import torch
from generativelib.config_tools.base import ConfigReader
from generativelib.dataset.base import BaseMaskProcessor
from generativelib.dataset.mask_processors import MASK_PROCESSORS
from generativelib.model.arch.enums import ModelTypes, GenerativeModules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.base import LOSSES, EvalItem, EvaluateProcessor
from generativelib.model.evaluators.enums import EvaluatorType, MetricName
from generativelib.model.train.base import Arch, ArchOptimizer, ArchOptimizersCollection, BaseTrainTemplate


class TrainConfigSerializer(ConfigReader):
    def __init__(self, config_folder_path: str, phase: ExecPhase):
        super().__init__(config_folder_path, phase)
        self._mask_processors: Dict = self.config['mask_processors']
        self._models: Dict = self.config['models']
    
    def serialize_mask_processors(self) -> List[BaseMaskProcessor]:
        processors: List[BaseMaskProcessor] = []
        
        for name, values in self._mask_processors.items():
            if values["enabled"]:
                cls = MASK_PROCESSORS.get(name)
                processors.append(cls(**values["params"]))
        
        return processors
    
    def serialize_evaluators(self, device: torch.device, eval_info: Dict) -> List[EvalItem]:
        evals = []
        for eval_name, eval_params in eval_info.items():
            exec_params = eval_params["execution"]
            init_params = eval_params["init"]
            
            eval_type = EvaluatorType.LOSS
            if eval_name in MetricName:
                eval_type = EvaluatorType.METRIC
            
            if exec_params['weight'] > 0.0:
                cls = LOSSES.get(eval_name)
                evaluator = EvalItem(
                    callable_fn=cls(**init_params).to(device),
                    name=eval_name,
                    type=eval_type,
                    exec_phase=exec_params["exec_phase"],
                    weight=exec_params["weight"]
                )
                
                evals.append(evaluator)
        return evals
    
    def serialize_model(self, device: torch.device, model_type: ModelTypes) -> tuple:
        model_dict: Dict = self._models.get(model_type.name)
        optimization_params: Dict = model_dict.get("optimization_params")
        model_params = model_dict.get("model_params")
        modules: Dict = model_dict.get("modules")
        arch_collect = ArchOptimizersCollection()
        
        arch_params = self.params_by_section(section='arch', keys=['in_ch', 'f_maps'])
        for m_name, module_info in modules.items():
            cls = GenerativeModules[m_name].value
            arch = Arch(GenerativeModules[m_name], cls(**arch_params).to(device))
            evals = self.serialize_evaluators(device, module_info.get("evaluators_info"))
            
            eval_proc = EvaluateProcessor(evals)
            arch_collect.append(ArchOptimizer(arch, eval_proc, optimization_params))
            
        return arch_collect, model_params