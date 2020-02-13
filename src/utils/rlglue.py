import numpy as np
from RlGlue import BaseAgent

def identity(s):
    return s

class OffPolicyWrapper(BaseAgent):
    def __init__(self, agent, gamma, b_policy, t_policy, observationChannel = identity):
        self.agent = agent
        self.gamma = gamma
        self.b_policy = b_policy
        self.t_policy = t_policy
        self.observationChannel = observationChannel
        self.s_t = None
        self.a_t = None
        self.obs_t = None

    def start(self, s):
        self.s_t = s
        self.obs_t = self.observationChannel(s)
        self.a_t = self.b_policy.selectAction(s)
        return self.a_t

    def step(self, r, s):
        gamma = self.gamma
        obs_tp1 = self.observationChannel(s)
        p = self.t_policy.ratio(self.b_policy, self.s_t, self.a_t)

        self.agent.update(self.obs_t, self.a_t, obs_tp1, r, gamma, p)

        self.s_t = s
        self.a_t = self.b_policy.selectAction(s)
        self.obs_t = obs_tp1

        return self.a_t

    def end(self, r):
        p = self.t_policy.ratio(self.b_policy, self.s_t, self.a_t)

        self.agent.update(self.obs_t, self.a_t, np.zeros_like(self.obs_t), r, 0, p)
        # self.agent.reset()
