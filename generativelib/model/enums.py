from enum import Enum


class ExecPhase(Enum):
    TRAIN = "Train"
    VALID = "Valid"
    TEST = "Test"
    INFER = "Infer"
    ANY = 'Any'