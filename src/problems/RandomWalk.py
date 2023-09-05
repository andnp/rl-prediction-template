import numpy as np
from PyRlEnvs.domains.RandomWalk import buildRandomWalk, invertedFeatures, tabularFeatures, dependentFeatures
from problems.BaseProblem import BaseProblem

from utils.representations import MappedRepresentation
from utils.policies import Policy

class RandomWalk(BaseProblem):
    def _buildRepresentation(self, name):
        m = None
        if name == 'tabular':
            m = tabularFeatures(self.states)

        if name == 'inverted':
            m = invertedFeatures(self.states)

        if name == 'dependent':
            m = dependentFeatures(self.states)

        assert m is not None
        return MappedRepresentation(m)

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        self.states = self.params['states']

        mu_pl = self.params['behavior']
        mu_probs = np.array([mu_pl, 1 - mu_pl])
        self.behavior = Policy(lambda s: mu_probs, rng=np.random.default_rng(self.seed))

        pi_pl = self.params['target']
        pi_probs = np.array([pi_pl, 1 - pi_pl])
        self.target = Policy(lambda s: pi_probs, rng=np.random.default_rng(self.seed))

        self.env = buildRandomWalk(self.states)(self.seed)

        # build representation
        representation = self.params['representation']
        self.rep = self._buildRepresentation(representation)

        self.observations = (self.rep.features(), )
        self.actions = 2
        self.gamma = 1.0

        self.agent = self.Agent(self.gamma, self.actions, self.params, self.rep, self.behavior, self.target)
