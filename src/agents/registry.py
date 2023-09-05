from agents.GTD2 import GTD2
from agents.HTD import HTD
from agents.TD import TD
from agents.TDC import TDC
from agents.TDRC import TDRC
from agents.VTrace import VTrace

def getAgent(name):
    if name == 'TD':
        return TD

    if name == 'VTrace':
        return VTrace

    if name == 'TDC':
        return TDC

    if name == 'TDRC':
        return TDRC

    if name == 'GTD2':
        return GTD2

    if name == 'HTD':
        return HTD

    raise NotImplementedError()
