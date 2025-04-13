import os
from torch import cuda
import albumentations as A
from src.preprocessing.processors import *

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
AUGMENTATIONS = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], p=0.6),
            A.RandomCrop(height=512, width=512, p=0.7),
        ])

PREPROCESSORS = [
            Binarize(),
            RotateMaskProcessor(),
            CropToContentProcessor(),
            EnchanceProcessor(kernel_size=3),
            AutoAdjust(),
            Unbinarize()
        ]

MASKS_FILE_EXTENSIONS = ['.png']