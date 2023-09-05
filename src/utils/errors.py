import numpy as np
from numba import njit

def matrix_power(X: np.ndarray, n: int):
    A = X
    for _ in range(n):
        A = np.dot(A, X)

    return A

def getSteadyStateDist(P: np.ndarray):
    A = matrix_power(P, 1000)
    d = np.mean(A, axis=0)
    return d

@njit(cache=True)
def partiallyApplyMSPBE(X: np.ndarray, P_gamma: np.ndarray, R: np.ndarray, db: np.ndarray):
    D = np.diag(db)
    I = np.eye(X.shape[0])

    A = X.T.dot(D).dot(I - P_gamma).dot(X)
    b = X.T.dot(D).dot(R)
    C = X.T.dot(D).dot(X)

    # go ahead and precompute this inverse too, it's expensive
    Cinv = np.linalg.pinv(C)

    return (A, b, Cinv)

@njit(cache=True)
def MSPBE(w: np.ndarray, A: np.ndarray, b: np.ndarray, Cinv: np.ndarray):
    dx = np.dot(-A, w) + b

    return dx.T.dot(Cinv).dot(dx)

@njit(cache=True)
def compute_vpi(P_gamma: np.ndarray, R: np.ndarray):
    I = np.eye(R.shape[0])

    return np.linalg.pinv(I - P_gamma).dot(R)

@njit(cache=True)
def MSVE(w: np.ndarray, X: np.ndarray, db: np.ndarray, vpi: np.ndarray):
    v_hat = X.dot(w)
    err = v_hat - vpi

    return db.dot(np.square(err))
