from enum import Enum, auto

from generativelib.model.arch.gan import GanDiscriminator, GanGenerator


class GenerativeModules(Enum):
    GAN_GENERATOR = GanGenerator
    GAN_DISCRIMINATOR = GanDiscriminator


class ModelTypes(Enum):
    GAN = auto()
    DIFFUSION = auto()