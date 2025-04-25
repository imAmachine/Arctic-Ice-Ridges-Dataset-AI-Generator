import os
import random
from torch import cuda
import torchvision.transforms.v2 as T
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
class OneOf(torch.nn.Module):
    def __init__(self, transforms, p=1.0):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            t = random.choice(self.transforms)
            return t(x)
        return x
    
class RandomRotate(torch.nn.Module):
    def __init__(self, angles=(90, 180, 270), p=0.8):
        super().__init__()
        self.angles = angles
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return T.functional.rotate(x, angle)
        return x
    
AUGMENTATIONS = T.Compose([
    OneOf([
        T.RandomCrop((1024, 1024)),
        T.RandomCrop((768, 768)),
        T.RandomCrop((512, 512)),
    ], p=1.0),
    RandomRotate(p=0.8),
    T.RandomHorizontalFlip(p=0.8),
    T.RandomVerticalFlip(p=0.8),
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