from typing import Any, Dict
import numpy as np
from numba import njit
from agents.BaseAgent import BaseAgent
from utils.policies import Policy
from utils.representations import Representation
from agents.optimizers.Optimizer import get_optimizer

@njit(cache=True)
def tdc_update(
    w: np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    xp: np.ndarray,
    r: float,
    gamma: float,
    rho: float,
    tau: float,
):
    vp = xp.dot(w)
    v = x.dot(w)

    delta_hat = x.dot(h)

    delta = r + gamma * vp - v
    dw = rho * delta * x - rho * gamma * tau * np.tanh(delta_hat / tau) * xp
    dh = rho * (delta - delta_hat) * x

    return -dw, -dh

class TDCHuber(BaseAgent):
    def __init__(self, gamma: float, actions: int, params: Dict[str, Any], rep: Representation, mu: Policy, pi: Policy):
        super().__init__(gamma, actions, params, rep, mu, pi)
        self.theta = np.zeros(self.features)
        self.h = np.zeros(self.features)

        Opt = get_optimizer(self.params['optimizer'])
        self.h_opt = Opt(self.features, {
            'alpha': self.params['alpha'] * self.params['eta'],
        })

        self.tau = self.params['tau']

    def update(self, x, a, xp, r, gamma, rho):
        dw, dh = tdc_update(self.theta, self.h, x, xp, r, gamma, rho, self.tau)
        self.theta = self.opt.apply(self.theta, dw)
        self.h = self.h_opt.apply(self.h, dh)

    def weights(self):
        return self.theta

    def set_weights(self, w: np.ndarray):
        self.theta = w
