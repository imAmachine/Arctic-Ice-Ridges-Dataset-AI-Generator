from typing import Dict, List
from generativelib.config_tools.base import ConfigReader
from generativelib.config_tools.default_values import ARCH_PARAMS_KEY, EVALS_KEY, MASK_PROCESSORS_KEY, MODEL_PARAMS_KEY, MODELS_KEY, MODULES_KEY, OPTIMIZER_KEY
from generativelib.dataset.base import BaseMaskProcessor
from generativelib.model.arch.enums import ModelTypes, GenerativeModules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.train.base import ArchModule, ModuleOptimizer, ModuleOptimizersCollection

# [AI] class
class InferenceConfigDeserializer(ConfigReader):
    def __init__(self, config_folder_path: str):
        super().__init__(config_folder_path, ExecPhase.TRAIN) # должно быть ExecPhase.INFER
        self._models: Dict = self.config[MODELS_KEY]

    def get_module_arch_info(self, model_type: GenerativeModules, module_name: str) -> Dict:
        modules = self._models[model_type.name.lower()][MODULES_KEY]
        return modules[module_name][ARCH_PARAMS_KEY]
    
    def create_arch_module(self, model_type: GenerativeModules, module_name: str) -> ArchModule:
        arch_info = self.get_module_arch_info(model_type, module_name)
        return ArchModule.create_from_dict(module_name, arch_info)

class TrainConfigDeserializer(ConfigReader):
    def __init__(self, config_folder_path: str, phase: ExecPhase):
        super().__init__(config_folder_path, phase)
        self._mask_processors: Dict = self.config[MASK_PROCESSORS_KEY]
        self._models: Dict = self.config[MODELS_KEY]
    
    def all_dataset_masks(self) -> List[BaseMaskProcessor]:
        processors: List[BaseMaskProcessor] = []
        
        for name, values in self._mask_processors.items():
            proc = BaseMaskProcessor.from_dict(name, values)
            if proc: processors.append(proc)
        
        return processors
    
    def all_eval_items(self, eval_info: Dict) -> List[EvalItem]:
        evals: List[EvalItem] = []
        
        for eval_name, eval_params in eval_info.items():
            eval_it = EvalItem.from_dict(eval_name, eval_params)
            
            if eval_it: evals.append(eval_it)
            
        return evals
    
    def module_optimizer(self, module_name: str, arch_info: Dict, evals_info: Dict, optim_info: Dict) -> ModuleOptimizer:
        arch_module = ArchModule.create_from_dict(module_name, arch_info)
        evals = self.all_eval_items(evals_info)
        module_optimizer = ModuleOptimizer.create(arch_module, evals, optim_info)
        
        return module_optimizer
    
    def optimize_collection(self, model_type: ModelTypes) -> ModuleOptimizersCollection:
        arch_collect = ModuleOptimizersCollection()
        
        modules = self._models.get(model_type.name.lower()).get(MODULES_KEY)
        
        for module_name, module_info in modules.items():
            arch_info = module_info.get(ARCH_PARAMS_KEY, {})
            evals_info = module_info.get(EVALS_KEY, {})
            optim_info = module_info.get(OPTIMIZER_KEY, {})
            
            module_optimizer = self.module_optimizer(module_name, arch_info, evals_info, optim_info)
            arch_collect.append(module_optimizer)
        
        return arch_collect
    
    def model_params(self, model_type: GenerativeModules):
        model_dict: Dict = self._models.get(model_type.name.lower())
        return model_dict.get(MODEL_PARAMS_KEY)