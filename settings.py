import os
from torch import cuda
from src.preprocessing.processors import *
from src.common.structs import ModelType as models, LossName as losses, MetricsName as metrics, ExecPhase as phases, EvaluatorType as eval_type

# путь к файлам с конфигарциями
CONFIG = './config.json'
DEFAULT_TRAIN_CONF = {
    "target_image_size": 256,
    "model_base_features": 64,
    "n_critic": 5,
    "evaluators_info": {
        models.GENERATOR.value: {
            losses.ADVERSARIAL.value: {
                "weight": 1.0,
                "exec_phase": phases.ANY.value,
                "type": eval_type.LOSS.value,
            },
            losses.BCE.value: {
                "weight": 0.1,
                "exec_phase": phases.ANY.value,
                "type": eval_type.LOSS.value,
            },
            losses.L1.value: {
                "weight": 1.0,
                "exec_phase": phases.ANY.value,
                "type": eval_type.LOSS.value,
            },
            metrics.PRECISION.value: {
                "weight": 1.0,
                "exec_phase": phases.ANY.value,
                "type": eval_type.METRIC.value,
            },
            metrics.F1.value: {
                "exec_phase": phases.ANY.value,
                "type": eval_type.METRIC.value,
            },
            metrics.IOU.value: {
                "exec_phase": phases.ANY.value,
                "type": eval_type.METRIC.value,
            }
        },
        models.DISCRIMINATOR.value: {
            losses.WASSERSTEIN.value: {
                "weight": 1.0,
                "exec_phase": phases.ANY.value,
                "type": eval_type.LOSS.value,
            },
            losses.GP.value: {
                "weight": 10.0,
                "exec_phase": phases.TRAIN.value,
                "type": eval_type.LOSS.value,
            },
        }
    },
    "optimization_params": {
        eval_type.METRIC.name: metrics.IOU.value,
        "mode": "max",
        "lr": 0.0005
    }
}

DEFAULT_TEST_CONF = {
} 

# путь к файлу с геоанализом исходных снимков
GEODATA_PATH = "./data/geo_data.csv"

# путь к корневой директории для обработанных данных
OUTPUT_FOLDER_PATH = "./data/processed_output/"

# пути к директориям для масок
MASKS_FOLDER_PATH = "./data/masks/" # исходные маски
PREPROCESSED_MASKS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "preprocessed") # предобработанные входные маски
AUGMENTED_DATASET_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'augmented_dataset') # обработанные
GENERATED_GAN_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'generated')

# пути к весам модели
WEIGHTS_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'model_weight/weights')
GENERATOR_PATH = os.path.join(WEIGHTS_PATH, 'generator.pth')

# ================ настройка основных параметров ==================
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

PREPROCESSORS = [
            RotateMaskProcessor(),
            Binarize(),
            CropToContentProcessor(),
            AutoAdjust(),
            Unbinarize()
        ]

MASKS_FILE_EXTENSIONS = ['.png']