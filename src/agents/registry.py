from agents.TD import TD
from agents.TDRC import TDRC

def getAgent(name):
    if name == 'TD':
        return TD

    if name == 'TDRC':
        return TDRC

    raise NotImplementedError()
