from enum import Enum


class ExecPhases(Enum):
    TRAIN = "train"
    EVAL = "eval"
    VALID = "valid"
    TEST = 'test'
    ANY = 'any'
    

class ModelTypes(Enum):
    GENERATOR = "gen"
    DISCRIMINATOR = "discr"
    DIFFUSION = "dif"


class LossNames(Enum):
    TOTAL = "Total"
    ADVERSARIAL = "Adversarial"
    BCE = "BCE"
    L1 = "L1"
    WASSERSTEIN = "Wasserstein"
    GP = "Gradient Penalty"