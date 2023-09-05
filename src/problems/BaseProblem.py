from typing import Optional
from PyRlEnvs.BaseEnvironment import BaseEnvironment

from agents.BaseAgent import BaseAgent
from agents.registry import getAgent
from experiment.ExperimentModel import ExperimentModel
from utils.representations import Representation

class BaseProblem:
    def __init__(self, exp: ExperimentModel, idx: int):
        self.exp = exp
        self.idx = idx

        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']
        self.seed = exp.getRun(idx)

        self.Agent = getAgent(exp.agent)

        self.agent: Optional[BaseAgent] = None
        self.env: Optional[BaseEnvironment] = None
        self.rep: Optional[Representation] = None

        self.features = 0
        self.actions = 0

    def getEnvironment(self):
        assert self.env is not None
        return self.env

    def getRepresentation(self):
        assert self.rep is not None
        return self.rep

    def getAgent(self):
        assert self.agent is not None
        return self.agent
