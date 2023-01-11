from problems.RandomWalk import RandomWalk
from problems.Baird import Baird

def getProblem(name):
    if name == 'RandomWalk':
        return RandomWalk

    if name == 'Baird':
        return Baird

    raise NotImplementedError()
