import RlGlue
import numpy as np
from typing import Any, Dict, Optional
from utils.representations import Representation
from utils.policies import Policy

class BaseAgent(RlGlue.BaseAgent):
    def __init__(self, gamma: float, actions: int, params: Dict[str, Any], rep: Representation, mu: Policy, pi: Policy):
        self._gamma = gamma
        self.params = params
        self.actions = actions

        self.rep = rep
        self.mu = mu
        self.pi = pi

        # track one-step lag
        self.s_t: Optional[np.ndarray] = None
        self.a_t: Optional[int] = None
        self.obs_t: Optional[np.ndarray] = None

    def start(self, s: np.ndarray):
        self.s_t = s
        self.obs_t = self.rep.encode(s)
        self.a_t = self.mu.selectAction(s)
        return self.a_t

    def step(self, r: float, s: np.ndarray):
        obs_tp1 = self.rep.encode(s)

        assert self.s_t is not None and self.a_t is not None and self.obs_t is not None
        rho = self.pi.ratio(self.mu, self.s_t, self.a_t)

        self.update(self.obs_t, self.a_t, obs_tp1, r, self._gamma, rho)

        self.s_t = s
        self.a_t = self.mu.selectAction(s)
        self.obs_t = obs_tp1
        return self.a_t

    def end(self, r: float):
        assert self.s_t is not None and self.a_t is not None and self.obs_t is not None
        rho = self.pi.ratio(self.mu, self.s_t, self.a_t)

        self.update(self.obs_t, self.a_t, np.zeros_like(self.obs_t), r, 0, rho)

        self.s_t = None
        self.obs_t = None
        self.a_t = None

    def update(self, x: np.ndarray, a: int, xp: np.ndarray, r: float, gamma: float, rho: float):
        ...

    def weights(self) -> np.ndarray:
        raise NotImplementedError()

    def set_weights(self, w: np.ndarray):
        raise NotImplementedError()
