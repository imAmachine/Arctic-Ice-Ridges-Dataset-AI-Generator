from src.preprocessing.processors import *
from src.common.enums import *
from src.dataset.strategies import *

DEFAULT_TRAIN_CONF = {
    "gan": {
        "target_image_size": 256,
        "model_base_features": 64,
        "n_critic": 5,
        "checkpoints_ratio": 25,
        "evaluators_info": {
            ModelType.GENERATOR.value: {
                LossName.ADVERSARIAL.value: {
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.BCE.value: {
                    "weight": 0.2,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.FOCAL.value: {
                    "weight": 0.4,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.DICE.value: {
                    "weight": 0.6,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.L1.value: {
                    "weight": 0.0,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                MetricName.PRECISION.value: {
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.METRIC.value,
                },
                MetricName.F1.value: {
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.METRIC.value,
                },
                MetricName.IOU.value: {
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.METRIC.value,
                }
            },
            ModelType.DISCRIMINATOR.value: {
                LossName.WASSERSTEIN.value: {
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.GP.value: {
                    "weight": 10.0,
                    "exec_phase": ExecPhase.TRAIN.value,
                    "type": EvaluatorType.LOSS.value,
                },
            }
        },
        "optimization_params": {
            EvaluatorType.METRIC.name: MetricName.IOU.value,
            "mode": "max",
            "lr": 0.0005
        },
        "mask_processors": {
            Padding.__name__: [{
                "ratio": 0.15,
            }, True],
            EllipsoidPadding.__name__: [{
                "ratio": 0.15,
            }, False],
            RandomHoles.__name__: [{
                "count": 1,
                "min_sz": 40,
                "max_sz": 60,
                "inversed": False
            }, False]
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