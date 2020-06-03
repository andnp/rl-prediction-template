from src.problems.RandomWalk import RandomWalk

def getProblem(name):
    if name == 'RandomWalk':
        return RandomWalk

    raise NotImplementedError()
