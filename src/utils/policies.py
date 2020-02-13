import numpy as np
from src.utils.random import sample

class Policy:
    def __init__(self, probs):
        self.probs = probs

    def selectAction(self, s):
        action_probabilities = self.probs(s)
        return sample(action_probabilities)

    def ratio(self, other, s, a):
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs):
    return Policy(lambda s: probs[s])

def fromActionArray(probs):
    return Policy(lambda s: probs)
