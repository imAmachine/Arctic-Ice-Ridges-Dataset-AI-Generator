from enum import Enum


class ExecPhase(Enum):
    TRAIN = "Train"
    VALID = "Valid"
    TEST = 'Test'
    ANY = 'Any'
    

class ModelType(Enum):
    GENERATOR = "Generator"
    DISCRIMINATOR = "Discriminator"
    GAN = "gan"
    DIFFUSION = "diffusion"


class EvaluatorType(Enum):
    LOSS = "Loss"
    METRIC = "Metric"


class LossName(Enum):
    TOTAL = "Total"
    ADVERSARIAL = "Adversarial"
    BCE_Logits = "BCE_Logits"
    BCE = "BCE"
    L1 = "L1"
    EDGE = "Edge"
    FOCAL = "Focal"
    DICE = "Dice"
    WASSERSTEIN = "Wasserstein"
    GP = "Gradient Penalty"


class MetricName(Enum):
    PRECISION = "Precision"
    F1 = "F1"
    IOU = "IoU"
    FD = "FD"