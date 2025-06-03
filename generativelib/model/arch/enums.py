from enum import Enum, auto

from generativelib.model.arch.gan import GanDiscriminator, GanGenerator
from generativelib.model.arch.diffusion import DiffusionUNet

class GenerativeModules(Enum):
    GAN_GENERATOR = GanGenerator
    GAN_DISCRIMINATOR = GanDiscriminator
    DIFFUSION = DiffusionUNet


class ModelTypes(Enum):
    GAN = auto()
    DIFFUSION = auto()