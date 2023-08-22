import os
import sys
import numpy as np
sys.path.append(os.getcwd())

import logging
import argparse
from PyExpUtils.utils.Collector import Collector, Ignore, Window, Subsample
from PyExpUtils.results.pandas import saveCollector
from PyRlEnvs.FiniteDynamics import FiniteDynamics

from RlGlue import RlGlue
from experiment import ExperimentModel

from problems.registry import getProblem
from utils.errors import partiallyApplyMSPBE, MSPBE, compute_vpi, MSVE

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--silent', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------

logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger('exp')
if not args.silent:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)


# ----------------------
# -- Experiment Def'n --
# ----------------------

exp = ExperimentModel.load(args.exp)
indices = args.idxs

for idx in indices:
    collector = Collector(
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'mspbe': Subsample(100),
            'msve': Window(100),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Ignore(),
    )
    collector.setIdx(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    agent = problem.getAgent()

    assert isinstance(env, FiniteDynamics)

    mu = problem.behavior
    pi = problem.target

    # takes actions according to mu and will pass the agent an importance sampling ratio
    # makes sure the agent only sees the state passed through rep.encode.
    # agent does not see raw state

    P_gamma = env.constructTransitionMatrix(pi.probs, gamma=problem.gamma)
    R = env.constructRewardVector(pi.probs)
    d = env.computeStateDistribution(mu.probs)

    X = np.array([
        rep.encode(i) for i in range(R.shape[0])
    ])

    # precompute matrices for cheaply computing MSPBE
    AbC = partiallyApplyMSPBE(X, P_gamma, R, d)
    vpi = compute_vpi(P_gamma, R)

    glue = RlGlue(agent, env)

    # Run the experiment
    glue.start()
    for step in range(exp.steps):
        # call agent.step and environment.step
        interaction = glue.step()
        collector.evaluate('mspbe', lambda: MSPBE(agent.weights(), *AbC))
        collector.evaluate('msve', lambda: MSVE(agent.weights(), X, d, vpi))

        # if terminal state, then restart the interface
        if interaction.t:
            glue.start()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
