from generativelib.model.evaluators.base import LOSSES

# enums
from generativelib.model.evaluators.enums import EvaluatorType, MetricName
from generativelib.model.arch.enums import GenerativeModules
from generativelib.model.enums import ExecPhase

from generativelib.preprocessing.processors import *
from generativelib.dataset.mask_processors import *

import inspect

DEFAULT_TRAIN_CONF = {
    "gan": {
        "target_image_size": 256,
        "model_base_features": 64,
        "n_critic": 5,
        "checkpoints_ratio": 25,
        "evaluators_info": {
            model.value: {
                k: {
                    "execution": {
                        "weight": 0.0,
                        "exec_phase": ExecPhase.TRAIN.value
                    },
                    "init": {
                        name: param.default
                        for name, param in inspect.signature(v.__init__).parameters.items()
                        if name != "self" and param.default is not inspect._empty
                    }
                }
                for k, v in LOSSES.items()
            }
            for model in [GenerativeModules.GENERATOR, GenerativeModules.DISCRIMINATOR]
        },
        "optimization_params": {
            EvaluatorType.METRIC.name: MetricName.IOU.value,
            "mode": "max",
            "lr": 0.0005
        },
        "mask_processors": {
            Padding.__name__: {
                "enabled": False,
                "params": {
                    "ratio": 0.15,
                }
            },
            EllipsoidPadding.__name__: {
                "enabled": False,
                "params": {
                    "ratio": 0.15,
                }
            },
            RandomWindow.__name__: {
                "enabled": True,
                "params": {
                    "window_size": 200,
                }
            },
            RandomHoles.__name__: {
                "enabled": False,
                "params": {
                    "count": 1,
                    "min_sz": 60,
                    "max_sz": 80,
                }
            }
        }
    },
}

DEFAULT_TEST_CONF = {
    "gan": {
        "Trainer": {
            "epochs": 30,
            "checkpoints_ratio": 15
        },
        "Dataset": {
            "augs_per_img": 5,
            "batch_size": 3,
            "val_ratio": 0.2,
            "workers": 4
        }
    }
} 