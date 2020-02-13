import numpy as np

def matrix_power(X, n):
    A = X
    for _ in range(n):
        A = np.dot(A, X)

    return A

def getSteadyStateDist(P):
    A = matrix_power(P, 1000)
    d = np.mean(A, axis=0)
    return d

def partiallyApplyMSPBE(X, P, R, db, gamma):
    D = np.diag(db)
    I = np.eye(X.shape[0])

    A = X.T.dot(D).dot(I - gamma * P).dot(X)
    b = X.T.dot(D).dot(R)
    C = X.T.dot(D).dot(X)

    # go ahead and precompute this inverse too, it's expensive
    Cinv = np.linalg.pinv(C)

    return (A, b, C, Cinv)

def MSPBE(w, A, b, C, Cinv):
    dx = np.dot(-A, w) + b

    return dx.T.dot(Cinv).dot(dx)
