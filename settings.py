import os
from torch import cuda
from src.preprocessing.processors import *
from src.common.structs import ModelTypes as models, LossNames as losses, MetricsNames as metrics

# путь к файлам с конфигарциями
CONFIG = './config.json'
DEFAULT_TRAIN_CONF = {
    "target_image_size": 256,
    "g_feature_maps": 64,
    "d_feature_maps": 32,
    "n_critic": 3,
    "losses_weights": {
        models.GENERATOR.value: {
            losses.ADVERSARIAL.value: 1,
            losses.BCE.value: 0.1,
            losses.L1.value: 1.0
        },
        models.DISCRIMINATOR.value: {
            losses.WASSERSTEIN.value: 1.0,
            losses.GP.value: 10.0
        }
    },
    "metrics_weights": {
        metrics.PRECISION.value: 1.0,
        metrics.F1.value: 1.0,
        metrics.IOU.value: 1.0
    },
    "optimization_params": {
        "metric": "iou",
        "mode": "max",
        "lr": 0.0005
    }
}

DEFAULT_TEST_CONF = {
    "augs_per_img": [1],
    "GenerativeModel": {
        "target_image_size": [256],
        "g_feature_maps": [32],
        "d_feature_maps": [32],
        "n_critic": [3],
        "losses_weights": {
            "gen": {
                "adv": [2.0],
                "bce": [1.0],
                "l1": [1.5]
            },
            "discr": {
                "gp": [10.0]
            }
        },
        "optimization_params": {
            "metric": "iou",
            "mode": "max",
            "lr": [0.0005]
        }
    },
    "GANTrainer": {
        "epochs": [100],
        "batch_size": [3],
        "val_ratio": [0.2]
    }
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
            Binarize(),
            RotateMaskProcessor(),
            CropToContentProcessor(),
            EnchanceProcessor(kernel_size=3),
            AutoAdjust(),
            Unbinarize()
        ]

MASKS_FILE_EXTENSIONS = ['.png']