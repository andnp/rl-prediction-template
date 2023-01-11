from typing import Any, Dict
import numpy as np
from numba import njit
from agents.BaseAgent import BaseAgent
from utils.policies import Policy
from utils.representations import Representation

@njit(cache=True)
def tdrc_update(
    w: np.ndarray,
    h: np.ndarray,
    alpha: float,
    x: np.ndarray,
    xp: np.ndarray,
    r: float,
    gamma: float,
    rho: float,
):
    vp = xp.dot(w)
    v = x.dot(w)

    delta_hat = x.dot(h)

    delta = r + gamma * vp - v
    dw = rho * delta * x - rho * gamma * delta_hat * xp
    dh = rho * (delta - delta_hat) * x - h

    w += alpha * dw
    h += alpha * dh
    return w, h

class TDRC(BaseAgent):
    def __init__(self, gamma: float, actions: int, params: Dict[str, Any], rep: Representation, mu: Policy, pi: Policy):
        super().__init__(gamma, actions, params, rep, mu, pi)
        self.features = rep.features()
        self.alpha = params['alpha']

        self.theta = np.zeros(self.features)
        self.h = np.zeros(self.features)

    def update(self, x, a, xp, r, gamma, rho):
        self.theta, self.h = tdrc_update(self.theta, self.h, self.alpha, x, xp, r, gamma, rho)

    def weights(self):
        return self.theta

    def set_weights(self, w: np.ndarray):
        self.theta = w
