'''Implements DWGLASSO and associated helper functions'''
import numba  # JIT compilation
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import combinations_with_replacement, repeat
from scipy.linalg import lu_solve, lu_factor


def periodogram_covar(x: np.array, y: np.array, tau: int, p: int):
    '''Takes in numpy arrays x and y, an int value tau for the lag and
    another int p for the maximum lag and returns the periodogram
    estimate of the covariance.
    '''
    assert np.allclose(x.mean(), 0), 'Signal x must be 0 mean!'
    assert np.allclose(y.mean(), 0), 'Signal y must be 0 mean!'
    T = len(x) - p
    if tau == 0:
        return (1 / T) * np.dot(x, y)
    else:
        return (1 / T) * np.dot(x[p:], y[p - tau:-tau])


# Exists essentially just to help implement the convenient notation in
# the function periodogram_covar_matrices.
class ColSelector(object):
    '''
    A helper object to select out the temperature data from our hdf store.
    '''
    def __init__(self, hdf: pd.io.pytables.HDFStore, keys, column):
        '''Takes in an HDFStore D, an iterable of keys, and the column we
        want to select from each of the key locations.  Precisely,
        each key location (hdf[keys[i]]) should contain a pd.DataFrame
        object D having the given column: D[column].

        If we have t = ColSelector(hdf, keys, column) then t.keys()
        will simply return keys, and t[k] will return
        hdf[k][column]
        '''
        self.keys = keys
        self.column = column
        self.hdf = hdf
        return

    def __getitem__(self, k):
        '''selector[k]'''
        return self.hdf[k][self.column]

    def keys(self):
        return self.keys


# Use the ColSelector class to give convenient access to an HDFStore
# e.g. give ColSelector the keys we want to iterate through, and the
# column we want to access.
def periodogram_covar_matrices(D, p: int):
    '''Makes use of the helper functions periodogram_covar to calculate
    the covariance matrices Rx(0) ... Rx(p) where x_i is the i'th
    column of D.  The matrices Rx(0) ... Rx(p - 1) form the top row
    of ZZT and Rx(1) ... Rx(p) form YZT.
    '''
    # ***<x, y> is O(n), and the data is copied to and from each process.
    # So, does parallelizing these calculations actually help?
    pool = Pool(p)  # calculating dot(x, y) for large T
    n = D.shape[1]
    Rx = np.zeros((n, n * (p + 1)))
    for ixi, jxj in combinations_with_replacement(
            enumerate(D.keys()), 2):
        i, xi = ixi
        j, xj = jxj
        xi = D[xi].values  # D[xi] should be a pd.Series.
        xj = D[xj].values
        Rx[i, j::n] = pool.starmap(periodogram_covar,
                                   zip(repeat(xi, p + 1),
                                       repeat(xj, p + 1),
                                       range(p + 1),
                                       repeat(p, p + 1))
                                   )
        # Fill in the rest by symmetry
        Rx[j, i::n] = Rx[i, j::n]
    pool.close()
    pool.join()
    Rx = list(np.split(Rx, p + 1, axis=1))
    return Rx


def form_ZZT(Rx: list[np.array]):
    '''Forms the matrix ZZT from the list of Rx matrices
    Rx = [Rx(0) Rx(1) ... Rx(p)] (n x n*(p + 1))

    ZZT is a np x np block toeplitz form from the 0 to p - 1 lagged
    covariance matrices of the n-vector x(t).
    '''
    ZZT_toprow = np.hstack(Rx[:-1])  # [Rx(0) ... Rx(p - 1)]
    del Rx  # Free up the memory
    n = ZZT_toprow.shape[0]
    p = ZZT_toprow.shape[1] / n
    ZZT = np.zeros((n * p, n * p))
    for tau in range(1, p + 1):
        ZZT[:, :n] = ZZT_toprow.T  # tau = 0
    for tau in range(1, p):  # Create the block toeplitz structure
        ZZT[:, tau * n:(tau + 1) * n] = np.roll(ZZT[:, (tau - 1) * n:tau * n],
                                                shift=n * tau, axis=0)
        ZZT[:n, tau * n:(tau + 1) * n] = ZZT_toprow[:, tau * n:(tau + 1) * n]
    return ZZT


def form_YZT(Rx: list[np.array]):
    '''Forms the matrix YZT from the list of Rx matrices
    Rx = [Rx(0) Rx(1) ... Rx(p)] (n x n*(p + 1))

    YZT is an n x np matrix [Rx(1) ... Rx(p)]
    '''
    YZT = np.hstack(Rx[1:])  # [Rx(1) ... Rx(p)]
    return YZT


def dwglasso(D, p: int, lmbda: float, alpha: float=0.05, mu: float=0.1,
             tol=1e-6, max_iter=100):
    '''Minimizes over B:

    1/(2T)||Y - BZ||_F^2 + lmbda[alpha||B||_F^2 + (1 - alpha)G_DW(B)

    via ADMM.  G_DW is the depth wise group regularizer

    \sum_{ij}||Bt_ij||_2 where Bt_ij is the p-vector (B(1)_ij ... B(p)_ij)
    e.g. the filter coefficients from xj -> xi.

    if lmbda = 0 we have simple ordinary least squares, and if alpha = 1
    then we have tikhonov regularization.  mu is a tuning parameter
    for ADMM convergence and unless lmbda = 0 or alpha = 1, we need mu > 0.
    '''
    assert alpha >= 0 and alpha <= 1, 'Required: alpha \in [0, 1]'
    assert lmbda >= 0, 'Required: lmbda >= 0'
    assert mu >= 0, 'Required: mu >= 0'  # 0 only if lmbda = 0 or alpha = 1

    # Proximity operators
    @numba.jit(nopython=False, cache=True)
    def proxf(V: np.array):
        '''proximity operator of ||Y - BX||_F^2 + lmbda*(1 - alpha)||B||_F^2.
        See DWGLASSO paper for details
        '''
        return (lu_solve(lu_piv, YZT.T + V.T / mu)).T

    @numba.jit(nopython=False, cache=True)
    def proxg(V: np.array):
        '''proximity operator of alpha*lmbda*sum_ij||B_ij||_2 See DWGLASSO
        paper for details
        '''
        n = V.shape[0]
        p = V.shape[1] / n
        P = np.empty((n, n * p))
        for i in range(n):
            for j in range(n):
                Vtij = V[i, j::n]
                Vtij_l2 = np.linalg.norm(Vtij, ord=2)
                if Vtij_l2 == 0:
                    P[i, j::n] = 0
                else:
                    r = lmbda * (1 - alpha) * mu / Vtij_l2
                    P[i, j::n] = max(0, 1 - r) * Vtij
        return P

    def rel_err(Bxk, Bzk):  # F-norm difference between Bx and Bz
        return (1 / (n ** 2)) * np.linalg.norm(Bxk - Bzk, 'f')

    Rx = periodogram_covar_matrices(D, p)
    ZZT = form_ZZT(Rx)
    YZT = form_YZT(Rx)
    n = ZZT.shape[0] / p

    if lmbda == 0:  # OLS
        lu_piv = lu_factor(ZZT.T, overwrite_a=True)
        B = lu_solve(lu_piv, YZT.T, overwrite_b=True, check_finite=True).T
        return B

    else:
        if alpha == 1:  # Tikhonov regularization with lmbda
            # R and subsequently YZT are overwritten in the lu solver
            R = (ZZT + lmbda * np.eye(n))  # Has nothing to do with Rx
            lu_piv = lu_factor(R.T, overwrite_a=True)
            B = lu_solve(lu_piv, YZT.T, overwrite_b=True, check_finite=True).T
            return B

        else:  # DWGLASSO
            assert mu > 0, 'Required: mu > 0 (unless lmbda = 0, or alpha = 1)'
            if alpha == 0:
                raise RuntimeWarning('We need alpha > 0 to guarantee'
                                     'convergence to the optimal B matrix')
            r = (1 + 2 * mu * lmbda * alpha) / mu
            R = (ZZT + r * np.eye(n))  # Has nothing to do with Rx
            Bz, Bu = np.zeros((n, n * p)), np.zeros((n, n * p))  # Init with 0s
            Bx = proxf(Bz)
            k = 0
            rel_err_k = rel_err(Bx, Bz)
            while rel_err_k > tol and k < max_iter:  # ADMM iterations
                k += 1
                Bx = proxf(Bz - Bu)
                Bz = proxg(Bx + Bu)
                Bu = Bu + Bx - Bz
                rel_err_k = rel_err(Bx, Bz)

            if k == max_iter:
                raise RuntimeWarning('Max iterations exceeded! '
                                     'err = %0.14f' % rel_err_k)
            return Bz
    return
