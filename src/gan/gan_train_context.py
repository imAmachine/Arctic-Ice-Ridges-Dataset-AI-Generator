import torchvision.transforms.v2 as T

from typing import cast

from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

from src.train_context import TrainContext
from src.gan.gan_templates import GanTemplate


class GanTrainContext(TrainContext):    
    def _add_model_evaluators(self, template: OptimizationTemplate) -> None:
        """Добавляет лоссы для генератора и дискриминатора"""
        gan_template = cast(GanTemplate, template)
        
        discriminator = gan_template.get_discr_optimizer().module
        optimizers_collection = template.model_optimizers
        
        optimizers_collection.add_evals({
            GenerativeModules.GAN_GENERATOR: [
                EvalItem(
                    GeneratorLoss(discriminator),
                    name=LossName.ADVERSARIAL.name,
                    type=EvaluatorType.LOSS,
                    weight=1.0
                )
            ],
            GenerativeModules.GAN_DISCRIMINATOR: [
                EvalItem(
                    WassersteinLoss(discriminator),
                    name=LossName.WASSERSTEIN.name,
                    type=EvaluatorType.LOSS,
                    weight=1.0
                ),
                EvalItem(
                    GradientPenalty(discriminator),
                    name=LossName.GRADIENT_PENALTY.name,
                    type=EvaluatorType.LOSS,
                    weight=10.0,
                    exec_phase=ExecPhase.TRAIN
                )
            ]
        })

    def _get_model_transform(self, img_size: int) -> T.Compose:
        transforms = get_common_transforms(cast(int, img_size))
        return transforms

    def _init_visualize_hook(self, template: OptimizationTemplate) -> VisualizeHook:
        gan_template = cast(GanTemplate, template)
        gen_func = gan_template.get_gen_optimizer().module
        return self._visualize_hook(gen_callable=gen_func)