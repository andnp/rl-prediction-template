import numpy as np

class TD:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.alpha = params['alpha']

        self.theta = np.zeros(features)

    def computeUpdate(self, x, a, xp, r, gamma, p):
        vp = np.dot(xp, self.theta)
        v = np.dot(x, self.theta)

        delta = r + gamma * vp - v

        return p * delta * x

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeUpdate(x, a, xp, r, gamma, p)

        self.theta = self.theta + self.alpha * dtheta