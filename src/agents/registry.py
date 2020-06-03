from src.agents.TD import TD

def getAgent(name):
    if name == 'TD':
        return TD

    raise NotImplementedError()
