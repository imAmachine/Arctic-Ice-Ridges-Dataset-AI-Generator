from generativelib.model.evaluators.enums import LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate
from generativelib.model.arch.enums import Modules
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

from src.train_context import InitLossData, TrainContext


class GanTrainContext(TrainContext):
    MODULE_LOSSES = {
        Modules.GAN_GENERATOR: [
            InitLossData(
                callable_type=GeneratorLoss,
                name=LossName.ADVERSARIAL.name,
                weight=1.0,
                exec_phase=ExecPhase.ANY,
                init={}
            ),
        ],
        Modules.GAN_DISCRIMINATOR: [
            InitLossData(
                callable_type=WassersteinLoss,
                name=LossName.WASSERSTEIN.name,
                weight=1.0,
                exec_phase=ExecPhase.ANY,
                init={}
            ),
            InitLossData(
                callable_type=GradientPenalty,
                name=LossName.GRADIENT_PENALTY.name,
                weight=10.0,
                exec_phase=ExecPhase.TRAIN,
                init={}
            ),
        ]
    }
    TARGET_LOSS_MODULE = Modules.GAN_DISCRIMINATOR