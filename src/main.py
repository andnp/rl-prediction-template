import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel

from RlGlue.environment import BaseEnvironment
from RlGlue.agent import BaseAgent
from src.utils.rlglue import OffPolicyWrapper

from src.problems.registry import getProblem
from src.utils.errors import partiallyApplyMSPBE, MSPBE
from src.utils.Collector import Collector

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

collector = Collector()
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    agent = problem.getAgent()

    mu = problem.behavior
    pi = problem.target

    # takes actions according to mu and will pass the agent an importance sampling ratio
    # makes sure the agent only sees the state passed through rep.encode.
    # agent does not see raw state
    agent_wrapper = OffPolicyWrapper(agent, problem.getGamma(), mu, pi, rep.encode)

    X = np.array([
        rep.encode(i) for i in range(env.states + 1)
    ])

    P = env.buildTransitionMatrix(pi)
    R = env.buildAverageReward(pi)
    d = env.getSteadyStateDist(mu)

    # precompute matrices for cheaply computing MSPBE
    AbC = partiallyApplyMSPBE(X, P, R, d, problem.getGamma())

    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    glue.start()
    for step in range(exp.steps):
        # call agent.step and environment.step
        r, o, a, t = glue.step()

        mspbe = MSPBE(agent.theta, *AbC)
        collector.collect('mspbe', mspbe)

        # if terminal state, then restart the interface
        if t:
            glue.start()

    # tell the collector to start a new run
    collector.reset()

mspbe_data = collector.getStats('mspbe')

import matplotlib.pyplot as plt
from src.utils.plotting import plot
fig, ax = plt.subplots(1)

plot(ax, mspbe_data)
ax.set_title('MSPBE')

plt.show()
exit()

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('mspbe_summary.npy'), mspbe_data)
