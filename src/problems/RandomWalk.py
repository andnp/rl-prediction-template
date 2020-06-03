import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.environments.RandomWalk import RandomWalk as RWEnv

from src.utils.policies import Policy

class RandomWalk(BaseProblem):
    def _buildRepresentation(self, name):
        if name == 'tabular':
            return Tabular(self.states)

        if name == 'inverted':
            return Inverted(self.states)

        if name == 'dependent':
            return Dependent(self.states)

        raise NotImplementedError('Unexpected representation name: ' + name)

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        self.states = self.params['states']

        mu_pl = self.params['behavior']
        self.behavior = Policy(lambda s: [mu_pl, 1 - mu_pl])

        pi_pl = self.params['target']
        self.target = Policy(lambda s: [pi_pl, 1 - pi_pl])

        self.env = RWEnv(self.states)

        # build representation
        representation = self.params['representation']
        self.rep = self._buildRepresentation(representation)

        # build agent
        self.agent = self.Agent(self.rep.features(), 2, self.params)

    def getGamma(self):
        return 1.0

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

# --------------------
# -- Representation --
# --------------------

class Inverted(BaseRepresentation):
    def __init__(self, N):
        m = np.ones((N, N)) - np.eye(N)

        self.map = np.zeros((N+1, N))
        self.map[:N] = (m.T / np.linalg.norm(m, axis=1)).T

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class Tabular(BaseRepresentation):
    def __init__(self, N):
        m = np.eye(N)

        self.map = np.zeros((N+1, N))
        self.map[:N] = m

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class Dependent(BaseRepresentation):
    def __init__(self, N):
        nfeats = int(np.floor(N/2) + 1)
        self.map = np.zeros((N+1, nfeats))

        idx = 0
        for i in range(nfeats):
            self.map[idx, 0:i+1] = 1
            idx += 1

        for i in range(nfeats-1, 0, -1):
            self.map[idx, -i:] = 1
            idx += 1

        self.map[:N] = (self.map[:N].T / np.linalg.norm(self.map[:N], axis=1)).T

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]
