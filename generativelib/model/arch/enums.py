from enum import Enum


class GenerativeModules(Enum):
    GENERATOR = "Generator"
    DISCRIMINATOR = "Discriminator"


class ModelTypes(Enum):
    GAN = "GAN"
    DIFFUSION = "Diffusion"