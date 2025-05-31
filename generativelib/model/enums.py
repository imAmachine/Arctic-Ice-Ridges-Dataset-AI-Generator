from enum import Enum, auto


class ExecPhase(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()
    INFER = auto()
    ANY = auto()