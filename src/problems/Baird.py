from PyRlEnvs.domains.BairdCounterexample import BairdCounterexample, behaviorPolicy, targetPolicy, representationMatrix, initialWeights
from problems.BaseProblem import BaseProblem
from utils.policies import Policy
from utils.representations import MappedRepresentation


class Baird(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)

        self.env = BairdCounterexample(self.seed)
        self.behavior = Policy(behaviorPolicy)
        self.target = Policy(targetPolicy)

        self.rep = MappedRepresentation(representationMatrix())

        self.gamma = 0.99
        self.actions = 2
        self.observations = (self.rep.features(), )

        self.agent = self.Agent(self.gamma, self.actions, self.params, self.rep, self.behavior, self.target)
        self.agent.set_weights(initialWeights())
