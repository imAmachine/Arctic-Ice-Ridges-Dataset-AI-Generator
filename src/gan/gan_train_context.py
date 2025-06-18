from generativelib.model.evaluators.enums import LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate
from generativelib.model.arch.enums import Modules
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

from src.train_context import TrainContext


class GanTrainContext(TrainContext):
    MODULE_LOSSES = {
        Modules.GAN_GENERATOR: [
            {
                "callable_type": GeneratorLoss,
                "name": LossName.ADVERSARIAL.name,
                "weight": 1.0,
                "exec_phase": ExecPhase.ANY
            }
        ],
        Modules.GAN_DISCRIMINATOR: [
            {
                "callable_type": WassersteinLoss,
                "name": LossName.WASSERSTEIN.name,
                "weight": 1.0,
                "exec_phase": ExecPhase.ANY
            },
            {
                "callable_type": GradientPenalty,
                "name": LossName.GRADIENT_PENALTY.name,
                "weight": 10.0,
                "exec_phase": ExecPhase.TRAIN
            }
        ]
    }
    TARGET_LOSS_MODULE = Modules.GAN_DISCRIMINATOR

    def _init_visualize_hook(self, template: OptimizationTemplate) -> VisualizeHook:
        optimizers_collection = template.optimizers
        gen_optimizer = optimizers_collection.by_type(Modules.GAN_GENERATOR)
        gen_func = None
        
        if gen_optimizer:
            gen_func = gen_optimizer.module
        
        if gen_func:
            return self._visualize_hook(gen_callable=gen_func)
        
        raise ValueError('genfunc is None')