import numpy as np
from RlGlue import BaseEnvironment
from src.utils.errors import getSteadyStateDist

LEFT = 0
RIGHT = 1

class RandomWalk(BaseEnvironment):
    def __init__(self, size):
        self.states = size
        self.state = size // 2

    def start(self):
        self.state = self.states // 2
        return self.state

    def step(self, a):
        reward = 0
        terminal = False
        sp = self.state + 2 * a - 1

        if sp == -1:
            sp = self.states # terminal state
            reward = -1
            terminal = True

        elif sp == self.states:
            sp = self.states # terminal state
            reward = 1
            terminal = True

        self.state = sp

        return (reward, sp, terminal)

    def buildTransitionMatrix(self, policy):
        # number of states of the chain, plus a terminal state
        P = np.zeros((self.states + 1, self.states + 1))

        pl, pr = policy.probs(0)
        P[0, 1] = pr
        P[0, self.states] = pl

        for i in range(1, self.states):
            pl, pr = policy.probs(i)
            P[i, i - 1] = pl
            P[i, i + 1] = pr

        return P

    def buildAverageReward(self, policy):
        # number of state in the chain, plus a terminal state
        R = np.zeros(self.states + 1)

        # probability of transition times the reward received.
        pl, _ = policy.probs(0)
        R[0] = pl * -1

        _, pr = policy.probs(self.states - 1)
        R[self.states - 1] = pr * 1

        return R

    def getSteadyStateDist(self, policy):
        # Transition matrix _without_ terminal absorbing state
        P = np.zeros((self.states, self.states))

        pl, pr = policy.probs(0)
        P[0, self.states // 2] = pl
        P[0, 1] = pr

        pl, pr = policy.probs(self.states - 1)
        P[self.states - 1, self.states - 2] = pl
        P[self.states - 1, self.states // 2] = pr

        for i in range(1, self.states - 1):
            pl, pr = policy.probs(i)
            P[i, i - 1] = pl
            P[i, i + 1] = pr

        # now include the terminal state so dimensions are consistent
        db = np.zeros(self.states + 1)
        db[:self.states] = getSteadyStateDist(P)
        return db
