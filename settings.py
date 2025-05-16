import os
from torch import cuda
from src.preprocessing.processors import *
from src.common.structs import *

# путь к файлам с конфигарциями
CONFIG = './config.json'
DEFAULT_TRAIN_CONF = {
    "gan": {
        "target_image_size": 256,
        "model_base_features": 64,
        "n_critic": 5,
        "evaluators_info": {
            ModelType.GENERATOR.value: {
                LossName.ADVERSARIAL.value: {
                    "weight": 1.0,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.BCE.value: {
                    "weight": 0.1,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                LossName.L1.value: {
                    "weight": 1.0,
                    "exec_phase": ExecPhase.ANY.value,
                    "type": EvaluatorType.LOSS.value,
                },
                MetricName.PRECISION.value: {
                    "weight": 1.0,
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
                    "weight": 1.0,
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
    },
    "dif": {
        
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
    RotateMask(),
    AdjustToContent(),
    Crop(k=0.5),
]

MASKS_FILE_EXTENSIONS = ['.png']