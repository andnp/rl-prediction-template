import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from PyExpUtils.utils.Collector import Collector
from PyExpUtils.results.backends.pandas import saveCollector
from PyRlEnvs.FiniteDynamics import FiniteDynamics

from RlGlue import RlGlue
from experiment import ExperimentModel

from problems.registry import getProblem
from utils.errors import partiallyApplyMSPBE, MSPBE

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <path/to/description.json> <idxs...>')
    sys.exit(1)

exp = ExperimentModel.load(sys.argv[1])
idxs = list(map(int, sys.argv[2:]))

for idx in idxs:
    collector = Collector(idx)
    collector.setSampleRate('mspbe', int(exp.steps // 2000))

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

    glue = RlGlue(agent, env)

    # Run the experiment
    glue.start()
    for step in range(exp.steps):
        # call agent.step and environment.step
        interaction = glue.step()
        collector.evaluate('mspbe', lambda: MSPBE(agent.weights(), *AbC))

        # if terminal state, then restart the interface
        if interaction.t:
            glue.start()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector)
