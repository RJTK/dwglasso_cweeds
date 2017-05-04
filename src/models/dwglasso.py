'''
Implements DWGLASSO and associated helper functions.  We will load
in the dT data from /data/interim/interim_data.hdf and then apply the
dwglasso algorithm on a subsequence of this data.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
import sys
import numba  # JIT compilation
import numpy as np
import warnings
from scipy.linalg import lu_solve, lu_factor
from src.conf import MAX_P, ZZT_FILE_PREFIX, YZT_FILE_PREFIX


def dwglasso(ZZT: np.array, YZT: np.array, p: int=1, lmbda: float=0.0,
             alpha: float=0.05, mu: float=0.1,
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
        p = V.shape[1] // n
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

    def rel_err(Bxk, Bzk):  # F-norm difference between Bx and Bz per entry
        return (1 / ((n * p) ** 2)) * np.linalg.norm(Bxk - Bzk, 'f')

    n = ZZT.shape[0] // p

    if lmbda == 0:  # OLS
        print('OLS')
        lu_piv = lu_factor(ZZT.T, overwrite_a=True)
        B = lu_solve(lu_piv, YZT.T, overwrite_b=True, check_finite=True).T
        return B

    else:
        if alpha == 1:  # Tikhonov regularization with lmbda
            print('L2 Regularization')
            # R and subsequently YZT are overwritten in the lu solver
            R = (ZZT + lmbda * np.eye(n))  # Has nothing to do with Rx
            lu_piv = lu_factor(R.T, overwrite_a=True)
            B = lu_solve(lu_piv, YZT.T, overwrite_b=True, check_finite=True).T
            return B

        else:  # DWGLASSO
            print('DWGLASSO')
            assert mu > 0, 'Required: mu > 0 (unless lmbda = 0, or alpha = 1)'
            if alpha == 0:
                warnings.warn('We need alpha > 0 to guarantee convergence, '
                              'to the optimal B matrix', RuntimeWarning)
            r = (1 + 2 * mu * lmbda * alpha) / mu
            R = (ZZT + r * np.eye(n))  # Has nothing to do with Rx
            lu_piv = lu_factor(R.T, overwrite_a=True)
            Bz, Bu = np.zeros((n, n * p)), np.zeros((n, n * p))  # Init with 0s
            Bx = proxf(Bz)
            k = 0
            rel_err_k = rel_err(Bx, Bz)
            while rel_err_k > tol and k < max_iter:  # ADMM iterations
                k += 1
                print('iter: ', k, '(1/(np)^2)||Bx - Bz||_F^2: ', rel_err_k,
                      end='\r')
                sys.stdout.flush()
                Bx = proxf(Bz - Bu)
                Bz = proxg(Bx + Bu)
                Bu = Bu + Bx - Bz
                rel_err_k = rel_err(Bx, Bz)
            print()  # Print out a newline
            if k >= max_iter:  # Should only ever reach k == max_iter
                warnings.warn('Max iterations exceeded! '
                              'rel_err = %0.14f' % rel_err_k, RuntimeWarning)
            return Bz


def main():
    p = 1
    assert p <= MAX_P and p >= 1, 'p must be in (1, MAX_P)!'
    ZZT = np.load(ZZT_FILE_PREFIX + str(p) + '.npy')
    YZT = np.load(YZT_FILE_PREFIX + str(p) + '.npy')
    B_hat = dwglasso(ZZT, YZT, p, lmbda=0.1, alpha=0.1, tol=1e-6,
                     mu=0.1)
    print(B_hat)
    print(np.sum(np.abs(B_hat) > 0))
    return B_hat


if __name__ == '__main__':
    main()
