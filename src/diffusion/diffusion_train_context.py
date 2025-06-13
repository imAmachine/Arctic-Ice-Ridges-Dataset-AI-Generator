import torchvision.transforms.v2 as T

from typing import cast

from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

from src.train_context import TrainContext
from src.diffusion.diffusion_templates import DiffusionTemplate


class DiffusionTrainContext(TrainContext):
    def _add_model_evaluators(self, template: OptimizationTemplate) -> None:
        """Добавляет лоссы для диффузионной модели"""
        dif_template = cast(DiffusionTemplate, template)
        
        diffusion = dif_template.get_dif_optimizer().module
        optimizers_collection = template.model_optimizers

        optimizers_collection.add_evals({
            GenerativeModules.DIFFUSION:
            [EvalItem(
                nn.MSELoss(diffusion), 
                name=LossName.MSE.name, 
                type=EvaluatorType.LOSS, 
                weight=1.0
            )]
        })

    def _get_model_transform(self, img_size: int) -> T.Compose:
        base_transforms = get_common_transforms(cast(int, img_size))
        transforms = T.Compose([
            base_transforms,
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        return transforms
    
    def _init_visualize_hook(self, template: OptimizationTemplate) -> VisualizeHook:
        dif_template = cast(DiffusionTemplate, template)
        gen_func = dif_template._generate_from_noise
        return self._visualize_hook(gen_callable=gen_func)