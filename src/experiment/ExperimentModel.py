import json
from typing import Dict
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

class ExperimentModel(ExperimentDescription):
    def __init__(self, d: Dict, path: str):
        super().__init__(d, path)
        self.agent: str = d['agent']
        self.problem: str = d['problem']
        self.steps: int = d['steps']

def load(path: str):
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp
