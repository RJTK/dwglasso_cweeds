'''Implements DWGLASSO and associated helper functions'''
import numba
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import combinations_with_replacement, repeat
from scipy.linalg import lu_solve, lu_factor


def periodogram_covar(x: np.array, y: np.array, tau: int, p: int):
    '''Takes in a numpy arrays x and y value p for the lag
    and returns the periodogram estimate of the covariance.
    '''
    assert np.allclose(x.mean(), 0), 'Signal x must be 0 mean!'
    assert np.allclose(y.mean(), 0), 'Signal y must be 0 mean!'
    T = len(x) - p
    if tau == 0:
        return (1/T)*np.dot(x, y)
    else:
        return (1/T)*np.dot(x[p:], y[p - tau:-tau])


# This should work if D is a pd.io.pytables.HDFStore object
# But I should probably pull out just a subset that fits in memory
def calculate_ZZT(D, p: int):
    '''Makes use of the helper functions periodogram_covar to calculate
    the np x np matrix ZZT.  This essentially just consists of a block
    toeplitz matrix of x(t)x(t - p)^T factors.
    '''
    # ***<x, y> is O(n), and the data is copied to and from each process
    pool = Pool(p)  # calculating dot(x, y) for large T
    n = D.shape[1]
    ZZT_toprow = np.zeros((n, n*p))
    # The numerical index and the vector
    # This provides each xi xj combination
    for ixi, jxj in combinations_with_replacement(
            enumerate(D.keys()), 2):
        # If D is an HDFStore, we may need to be careful what keys we use
        i, xi = ixi
        j, xj = jxj
        xi = D[xi].values
        xj = D[xj].values
        ZZT_toprow[i, j::n] = pool.starmap(periodogram_covar,
                                           zip(repeat(xi, p),
                                               repeat(xj, p),
                                               range(p),
                                               repeat(p, p)))
        # Fills in the rest by symmetry
        ZZT_toprow[j, i::n] = ZZT_toprow[i, j::n]
    pool.close()
    pool.join()

    ZZT = np.zeros((n*p, n*p))
    ZZT[:, :n] = ZZT_toprow.T  # tau = 0
    for tau in range(1, p):  # Create the block toeplitz structure
        ZZT[:, tau*n:(tau + 1)*n] = np.roll(ZZT[:, (tau - 1)*n:tau*n],
                                            shift=n*tau, axis=0)
        ZZT[:n, tau*n:(tau + 1)*n] = ZZT_toprow[:, tau*n:(tau + 1)*n]
    return ZZT


def calculate_YZT(D, p: int):
    return


def dwglasso(D, p: int, lmbda: float, alpha: float, mu: float):
    '''
    '''
    ZZT = calculate_ZZT(D, p)
    n = ZZT.shape[0] / p
    r = (1 + 2*mu*lmbda*alpha)/mu
    R = (ZZT + r*np.eye(n)).T
    PLU = lu_factor(R, overwrite_a=True)  # Care, R is overwritten

    @numba.jit(nopython=False, cache=True)
    def proxf(V: np.array):
        '''proximity operator of ||Y - BX||_F^2 + lmbda*(1 - alpha)||B||_F^2.
        See DWGLASSO paper for details
        '''
        return (lu_solve(PLU, ZYT + V.T / mu)).T
        return

    @numba.jit(nopython=False, cache=True)
    def proxg(V: np.array):
        '''proximity operator of alpha*lmbda*sum_ij||B_ij||_2 See DWGLASSO
        paper for details
        '''
        n = V.shape[0]
        p = V.shape[1] / n
        P = np.empty((n, n*p))
        for i in range(n):
            for j in range(n):
                Vtij = V[i, j::n]
                Vtij_l2 = np.linalg.norm(Vtij, ord=2)
                if Vtij_l2 == 0:
                    P[i, j::n] = 0
                else:
                    r = lmbda*(1 - alpha)*mu / Vtij_l2
                    P[i, j::n] = max(0, 1 - r)*Vtij
        return P
    return
