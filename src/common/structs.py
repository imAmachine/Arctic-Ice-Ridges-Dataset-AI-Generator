from enum import Enum


class TrainPhases(Enum):
    TRAIN = "train"
    EVAL = "eval"
    VALID = "valid"
    TEST = 'test'