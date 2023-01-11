import numpy as np
from typing import Any, Callable
from PyExpUtils.utils.random import sample

class Policy:
    def __init__(self, probs: Callable[[Any], np.ndarray]):
        self.probs = probs

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(action_probabilities)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: np.ndarray):
    return Policy(lambda s: probs[s])

def fromActionArray(probs: np.ndarray):
    return Policy(lambda s: probs)
