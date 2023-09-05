import numpy as np
import numba as nb
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
        w, m, v = _adam_update(
            weights,
            updates,
            self.m,
            self.v,
            self._alpha,
            self._beta1,
            self._beta2,
            1e-8,
        )
        self.m = m
        self.v = v
        return w

@nb.njit(cache=True)
def _adam_update(w, u, m, v, alpha, beta1, beta2, eps):
    m = beta1 * m + (1 - beta1) * u
    v = beta2 * v + (1 - beta2) * (u ** 2)
    w = w - alpha * m / np.sqrt(v + eps)
    return w, m, v


def get_optimizer(name: str) -> Type[Optimizer]:
    if name == 'sgd':
        return SGD

    elif name == 'adam':
        return ADAM

    else:
        raise Exception()
