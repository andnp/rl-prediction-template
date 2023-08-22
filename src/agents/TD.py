from typing import Any, Dict
import numpy as np
from numba import njit
from agents.BaseAgent import BaseAgent
from utils.policies import Policy
from utils.representations import Representation

@njit(cache=True)
def td_update(
    w: np.ndarray,
    x: np.ndarray,
    xp: np.ndarray,
    r: float,
    gamma: float,
    rho: float,
):
    vp = xp.dot(w)
    v = x.dot(w)

    delta = r + gamma * vp - v
    dw = rho * delta * x

    return -dw

class TD(BaseAgent):
    def __init__(self, gamma: float, actions: int, params: Dict[str, Any], rep: Representation, mu: Policy, pi: Policy):
        super().__init__(gamma, actions, params, rep, mu, pi)
        self.theta = np.zeros(self.features)

    def update(self, x, a, xp, r, gamma, rho):
        dw = td_update(self.theta, x, xp, r, gamma, rho)
        self.theta = self.opt.apply(self.theta, dw)

    def weights(self):
        return self.theta

    def set_weights(self, w: np.ndarray):
        self.theta = w
