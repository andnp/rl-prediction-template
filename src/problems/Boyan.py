import numpy as np
from PyRlEnvs.domains.BoyanChain import BoyanChain, representationMatrix, behaviorPolicy
from problems.BaseProblem import BaseProblem

from utils.representations import MappedRepresentation
from utils.policies import Policy

class Boyan(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        self.behavior = Policy(behaviorPolicy, rng=np.random.default_rng(self.seed))
        self.target = Policy(behaviorPolicy, rng=np.random.default_rng(self.seed))

        self.env = BoyanChain(self.seed)

        # build representation
        self.rep = MappedRepresentation(representationMatrix())

        self.observations = (self.rep.features(), )
        self.actions = 2
        self.gamma = 1.0

        self.agent = self.Agent(self.gamma, self.actions, self.params, self.rep, self.behavior, self.target)
