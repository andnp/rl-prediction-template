import numpy as np
from typing import Dict, Type

class Optimizer:
    def __init__(self, features: int, params: Dict):
        self._features = features
        self._params = params

    def apply(self, weights: np.ndarray, updates: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, features: int, params: Dict):
        super().__init__(features, params)

        self._alpha = params['alpha']

    def apply(self, weights: np.ndarray, updates: np.ndarray) -> np.ndarray:
        return weights - self._alpha * updates

class ADAM(Optimizer):
    def __init__(self, features: int, params: Dict):
        super().__init__(features, params)

        self.m = np.zeros(features)
        self.v = np.zeros(features)

        self._alpha = params['alpha']
        self._beta1 = params.get('beta1', 0.9)
        self._beta2 = params.get('beta2', 0.999)

    def apply(self, weights: np.ndarray, updates: np.ndarray) -> np.ndarray:
        self.m = self._beta1 * self.m + (1 - self._beta1) * updates
        self.v = self._beta2 * self.v + (1 - self._beta2) * (updates ** 2)
        return weights - self._alpha * self.m / np.sqrt(self.v + 1e-8)


def get_optimizer(name: str) -> Type[Optimizer]:
    if name == 'sgd':
        return SGD

    elif name == 'adam':
        return ADAM

    else:
        raise Exception()
