from enum import Enum, auto


class EvaluatorType(Enum):
    LOSS = auto()
    METRIC = auto()


class LossName(Enum):
    TOTAL = auto()
    ADVERSARIAL = auto()
    BCE_Logits = auto()
    BCE = auto()
    MSE = auto()
    L1 = auto()
    EDGE = auto()
    FOCAL = auto()
    DICE = auto()
    WASSERSTEIN = auto()
    GRADIENT_PENALTY = auto()


class MetricName(Enum):
    PRECISION = auto()
    F1 = auto()
    IOU = auto()
    FRACTAL_DIMENSION = auto()