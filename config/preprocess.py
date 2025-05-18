from src.preprocessing.processors import *


PREPROCESSORS = [
    RotateMask(),
    AdjustToContent(),
    Crop(k=0.5),
]

MASKS_FILE_EXTENSIONS = ['.png']