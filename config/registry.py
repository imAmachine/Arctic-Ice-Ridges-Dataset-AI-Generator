from src.dataset.strategies import *


MASK_PROCESSORS = {
    Padding.__name__: Padding,
    EllipsoidPadding.__name__: EllipsoidPadding,
    RandomHoles.__name__: RandomHoles,
}