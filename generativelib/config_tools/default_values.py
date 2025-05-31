from generativelib.model.evaluators.base import LOSSES

# enums
from generativelib.model.evaluators.enums import MetricName
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase

from generativelib.preprocessing.processors import *
from generativelib.dataset.mask_processors import *


def get_default_train_conf():
    global_params = {
        "arch": {
            "image_size": 256,
            "in_ch": 1,
            "f_maps": 64,
        },
        "train": {
            "epochs": 1000,
            "checkpoint_ratio": 25,
        },
        "dataset": {
            "batch_size": 9,
            "augs": 15,
            "validation_size": 0.2,
            "shuffle": True,
            "workers": 4
        },
        "paths": {
            "masks": "./data/masks",
            "dataset": "./data/preprocessed",
            "processed": "./data/processed",
            "vizualizations": "./data/processed/vizualizations",
            "weights": "./data/processed/weights"
        }
    }
    mask_processors = {
        "Padding": {
            "enabled": False,
            "params": {
                "ratio": 0.15,
            }
        },
        "EllipsoidPadding": {
            "enabled": False,
            "params": {
                "ratio": 0.15,
            }
        },
        "RandomWindow": {
            "enabled": True,
            "params": {
                "window_size": 200,
            }
        },
        "RandomHoles": {
            "enabled": False,
            "params": {
                "count": 1,
                "min_sz": 60,
                "max_sz": 80,
            }
        }
    }
    models = {}
    
    models[ModelTypes.GAN.name] = {
        "model_params": {
            "n_critic": 5,
        },
        "optimization_params": {
            "evaluator": MetricName.IOU.name,
            "mode": "max",
            "lr": 0.0005,
            "betas": [0.0, 0.9],
        },
        "modules": {}
    }
        
    for module in GenerativeModules:
        models[ModelTypes.GAN.name]["modules"][module.name] = {
            "evaluators_info": {
                loss_name: {
                    "execution": {
                        "weight": 0.0,
                        "exec_phase": ExecPhase.TRAIN.name
                    },
                    "init": {}
                }
                for loss_name in LOSSES
            }
        }
    return {
        "global_params": global_params,
        "mask_processors": mask_processors,
        "models": models
    }

def get_default_test_conf():
    return {
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
    
def get_default_conf(phase: ExecPhase):
    if phase == ExecPhase.TRAIN:
        return get_default_train_conf()
    
    if phase == ExecPhase.TEST:
        return get_default_test_conf()
