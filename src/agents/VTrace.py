from typing import Any, Dict
import numpy as np
from numba import njit
from agents.BaseAgent import BaseAgent
from utils.policies import Policy
from utils.representations import Representation

@njit(cache=True)
def vtrace_update(
    w: np.ndarray,
    x: np.ndarray,
    xp: np.ndarray,
    r: float,
    gamma: float,
    rho: float,
):
    vp = xp.dot(w)
    v = x.dot(w)

    rho_hat = min(rho, 1)

    delta = r + gamma * vp - v
    dw = rho_hat * delta * x

    return -dw

class VTrace(BaseAgent):
    def __init__(self, gamma: float, actions: int, params: Dict[str, Any], rep: Representation, mu: Policy, pi: Policy):
        super().__init__(gamma, actions, params, rep, mu, pi)
        self.theta = np.zeros(self.features)

    def update(self, x, a, xp, r, gamma, rho):
        dw = vtrace_update(self.theta, x, xp, r, gamma, rho)
        self.theta = self.opt.apply(self.theta, dw)

    def weights(self):
        return self.theta

    def set_weights(self, w: np.ndarray):
        self.theta = w
