from src.common.enums import LossName, MetricName
from src.dataset.strategies import *
import torch.nn as nn
from src.models.gan.evaluators import *


MASK_PROCESSORS = {
    Padding.__name__: Padding,
    EllipsoidPadding.__name__: EllipsoidPadding,
    RandomHoles.__name__: RandomHoles,
    RandomWindow.__name__: RandomWindow,
}

LOSSES = {
    LossName.BCE_Logits.value: nn.BCEWithLogitsLoss,
    LossName.BCE.value: nn.BCELoss,
    LossName.L1.value: nn.L1Loss,
    LossName.EDGE.value: EdgeLoss,
    LossName.FOCAL.value: FocalLoss,
    LossName.DICE.value: DiceLoss
}

METRICS = {
    MetricName.PRECISION.value: precision_score,
    MetricName.F1.value: f1_score,
    MetricName.IOU.value: jaccard_score,
    MetricName.FD.value: FractalMetric,
}