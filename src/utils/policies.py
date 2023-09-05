import numpy as np
from typing import Any, Callable
from PyExpUtils.utils.random import sample

class Policy:
    def __init__(self, probs: Callable[[Any], np.ndarray], rng: np.random.Generator):
        self.rng = rng
        self.probs = probs

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(action_probabilities, self.rng)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: np.ndarray, rng: np.random.Generator):
    return Policy(lambda s: probs[s], rng)

def fromActionArray(probs: np.ndarray, rng: np.random.Generator):
    return Policy(lambda s: probs, rng)
