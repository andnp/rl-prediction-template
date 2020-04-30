import numpy as np

def matrix_power(X: np.ndarray, n: int):
    A = X
    for _ in range(n):
        A = np.dot(A, X)

    return A

def getSteadyStateDist(P: np.ndarray):
    A = matrix_power(P, 1000)
    d = np.mean(A, axis=0)
    return d

def partiallyApplyMSPBE(X: np.ndarray, P: np.ndarray, R: np.ndarray, db: np.ndarray, gamma: float):
    D = np.diag(db)
    I = np.eye(X.shape[0])

    A = X.T.dot(D).dot(I - gamma * P).dot(X)
    b = X.T.dot(D).dot(R)
    C = X.T.dot(D).dot(X)

    # go ahead and precompute this inverse too, it's expensive
    Cinv = np.linalg.pinv(C)

    return (A, b, C, Cinv)

def MSPBE(w: np.ndarray, A: np.ndarray, b: np.ndarray, C: np.ndarray, Cinv: np.ndarray):
    dx = np.dot(-A, w) + b

    return dx.T.dot(Cinv).dot(dx)
