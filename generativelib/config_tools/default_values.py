from generativelib.model.evaluators.base import LOSSES

# enums
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase

from generativelib.preprocessing.processors import *
from generativelib.dataset.mask_processors import *


# Константы для ключей
MASK_PROCESSORS_KEY = "mask_processors"
MODELS_KEY = "models"
GLOBAL_PARAMS_KEY = "global_params"
OPTIMIZER_KEY = "optimizer"
MODEL_PARAMS_KEY = "model_params"
MODULES_KEY = "modules"
DATASET_KEY = "dataset"
ARCH_PARAMS_KEY = "arch"
EVALS_KEY = "evals"
EXEC_PHASE_KEY = "exec_phase"
PATH_KEY = "path"
EXECUTION_KEY = "execution"
INIT_KEY = "init"


def get_default_train_conf():
    global_params = {
        ExecPhase.TRAIN.name.lower(): {
            "epochs": 1000,
            "visualize_interval": 5,
            "checkpoint_interval": 25,
        },
        DATASET_KEY: {
            "img_size": 256,
            "batch_size": 9,
            "augs": 30,
            "validation_size": 0.2,
            "shuffle": True,
            "workers": 4,
            "is_trg_masked": False
        },
        PATH_KEY: {
            "masks": "./data/masks",
            "dataset": "./data/preprocessed",
            "processed": "./data/processed",
            "interfaces": "./src/gui/interfaces/",
            "processed": "./data/processed/"
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
                "count": 2,
                "min_sz": 40,
                "max_sz": 50,
            }
        }
    }
    models = {}
    
    models[ModelTypes.GAN.name.lower()] = {
        MODEL_PARAMS_KEY: {
            "n_critic": 5,
        },
        MODULES_KEY: {}
    }

    models[ModelTypes.DIFFUSION.name.lower()] = {
        MODEL_PARAMS_KEY: {
            "num_timesteps": 1000
        },
        MODULES_KEY: {}
    }
    
    losses = {
        loss_name: {
            EXECUTION_KEY: {
                "weight": 0.0,
                EXEC_PHASE_KEY: ExecPhase.TRAIN.name
            },
            INIT_KEY: {}
        }
        for loss_name in LOSSES
    }
    
    for module in GenerativeModules:
        if "gan" in module.name.lower():
            models[ModelTypes.GAN.name.lower()][MODULES_KEY][module.name.lower()] = {
                ARCH_PARAMS_KEY: {
                    "in_ch": 1,
                    "f_base": 32,
                },
                EVALS_KEY: losses,
                OPTIMIZER_KEY: {
                    "type": "adam",
                    "params": {
                        "lr": 0.0005,
                        "betas": [0.0, 0.9]
                    }
                },
            }

        if "diffusion" in module.name.lower():
            models[ModelTypes.DIFFUSION.name.lower()][MODULES_KEY][module.name.lower()] = {
                ARCH_PARAMS_KEY: {
                    "in_ch": 1,
                    "f_base": 32,
                },
                EVALS_KEY: losses,
                OPTIMIZER_KEY: {
                    "type": "adam",
                    "params": {
                        "lr": 0.0005,
                        "betas": [0.0, 0.9]
                    }
                },
            }
    return {
        GLOBAL_PARAMS_KEY: global_params,
        MASK_PROCESSORS_KEY: mask_processors,
        MODELS_KEY: models
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
