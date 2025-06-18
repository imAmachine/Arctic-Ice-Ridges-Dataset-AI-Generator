from typing import Any, Dict, List, Union
from generativelib.config_tools.base import ConfigReader
from generativelib.config_tools.default_values import ARCH_PARAMS_KEY, LOSSES_KEY, MASK_PROCESSORS_KEY, MODEL_PARAMS_KEY, MODELS_KEY, MODULES_KEY, OPTIMIZER_KEY
from generativelib.dataset.base import MaskProcessor
from generativelib.model.arch.enums import ModelTypes, Modules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.base import LossItem
from generativelib.model.train.base import ArchModule, ModuleOptimizer, ModuleOptimizersCollection


# [AI] class
class InferenceConfigDeserializer(ConfigReader):
    def __init__(self, config_folder_path: str):
        super().__init__(config_folder_path, ExecPhase.TRAIN)
        self._models: Dict = self.config[MODELS_KEY]

    def get_module_arch_info(self, model_type: Modules, module_name: str) -> Dict:
        modules = self._models[model_type.name.lower()][MODULES_KEY]
        return modules[module_name][ARCH_PARAMS_KEY]
    
    def create_arch_module(self, model_type: Modules, module_name: str) -> ArchModule:
        arch_info = self.get_module_arch_info(model_type, module_name)
        return ArchModule.cls_from_dict(module_name, arch_info)


class TrainConfigDeserializer(ConfigReader):
    def __init__(self, config_folder_path: str, phase: ExecPhase):
        super().__init__(config_folder_path, phase)
        self._mask_processors: Dict[str, Any] = self.config.get(MASK_PROCESSORS_KEY, {})
        self._models: Dict[str, Any] = self.config.get(MODELS_KEY, {})

    def all_dataset_masks(self) -> List[MaskProcessor]:
        processors: List[MaskProcessor] = []

        for name, values in self._mask_processors.items():
            proc = MaskProcessor.from_dict(name, values)
            if proc:
                processors.append(proc)

        return processors

    def all_eval_items(self, eval_info: Dict[str, Any]) -> List[LossItem]:
        evals: List[LossItem] = []

        for eval_name, eval_params in eval_info.items():
            eval_it = LossItem.from_dict(eval_name, eval_params)
            if eval_it:
                evals.append(eval_it)

        return evals

    def module_optimizer(
        self,
        module_name: str,
        arch_info: Dict[str, Any],
        evals_info: Dict[str, Any],
        optim_info: Dict[str, Any]
    ) -> ModuleOptimizer:
        arch_module = ArchModule.cls_from_dict(module_name, arch_info)
        evals = self.all_eval_items(evals_info)
        return ModuleOptimizer.create(arch_module, evals, optim_info)

    def optimize_collection(self, model_type: ModelTypes) -> ModuleOptimizersCollection:
        arch_collect = ModuleOptimizersCollection()

        model_key = model_type.name.lower()
        model_entry = self._models.get(model_key, {})
        modules = model_entry.get(MODULES_KEY, {})

        for module_name, module_info in modules.items():
            arch_info = module_info.get(ARCH_PARAMS_KEY, {})
            evals_info = module_info.get(LOSSES_KEY, {})
            optim_info = module_info.get(OPTIMIZER_KEY, {})

            module_optimizer = self.module_optimizer(module_name, arch_info, evals_info, optim_info)
            arch_collect.append(module_optimizer)

        if len(arch_collect) == 0:
            raise ValueError('Коллекция оптимизаторов пуста')
        
        return arch_collect

    def model_params(self, model_type: Union[ModelTypes, Modules]) -> Dict[str, Any]:
        """Возвращает параметры модели (глобальные параметры инициализации, не оптимизаторы)."""
        model_key = model_type.name.lower()
        model_entry = self._models.get(model_key, {})
        return model_entry.get(MODEL_PARAMS_KEY, {})