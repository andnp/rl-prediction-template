from problems.Baird import Baird
from problems.Boyan import Boyan
from problems.RandomWalk import RandomWalk

def getProblem(name):
    if name == 'RandomWalk':
        return RandomWalk

    if name == 'Baird':
        return Baird

    if name == 'Boyan':
        return Boyan

    raise NotImplementedError()
