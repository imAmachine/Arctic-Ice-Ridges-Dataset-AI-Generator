from typing import cast
from generativelib.model.arch.enums import Modules
from generativelib.model.enums import ExecPhase
from generativelib.model.evaluators.enums import LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import OptimizationTemplate
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

from src.train_context import InitLossData, TrainContext
from src.diffusion.diffusion_templates import DiffusionTemplate


class DiffusionTrainContext(TrainContext):
    MODULE_LOSSES = {
        Modules.DIFFUSION: [
            InitLossData(
                callable_type=nn.MSELoss,
                name=LossName.MSE.name,
                weight=1.0,
                exec_phase=ExecPhase.ANY,
                init={"reduction": "mean"}
            )
        ],
    }
    TARGET_LOSS_MODULE = Modules.DIFFUSION
    
    def _init_visualize_hook(self, template: OptimizationTemplate) -> VisualizeHook:
        dif_template = cast(DiffusionTemplate, template)
        gen_func = dif_template._generate_from_noise
        return self._visualize_hook(gen_callable=gen_func)