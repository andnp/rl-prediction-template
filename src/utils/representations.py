import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation


class Representation(BaseRepresentation):
    def features(self) -> int:
        ...

class MappedRepresentation(Representation):
    def __init__(self, m: np.ndarray):
        self.map = addDummyTerminalState(m)

    def encode(self, s: int):
        return self.map[s]

    def features(self):
        return self.map.shape[1]


def addDummyTerminalState(m: np.ndarray):
    t = np.zeros((1, m.shape[1]))
    return np.concatenate([m, t], axis=0)
