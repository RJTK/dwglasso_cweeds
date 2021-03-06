'''
Implements DWGLASSO and associated helper functions.  We will load
in the dT data from /data/interim/interim_data.hdf and then apply the
dwglasso algorithm on a subsequence of this data.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
import time
import sys
import numba  # JIT compilation
import numpy as np
import warnings
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve, eigh
import matplotlib as mpl; mpl.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
from src.conf import MAX_P, ZZT_FILE_PREFIX, YZT_FILE_PREFIX,\
    X_VALIDATE_FILE_PREFIX


def build_YZ(X, p):
    '''
    Builds the Y (output) and Z (input) matrices for the model from
    the X matrix consisting of temperatures series in it's rows.
    We need to also provide the lag length of the model, p.

    X: n x T matrix of data.  Each row is a time series
      X = [x(0), x(1), ..., x(T - 1)]
    p: Model lag length

    Returns (Y, Z):
    Y: n x (T - p) matrix [x(p), x(p + 1), ..., x(T - 1)]
    Z: np x (T - p) matrix [z(p - 1), z(p), ..., z(T - 2)]
    where z(t) = [x(t).T x(t - 1).T, ..., x(t - p + 1).T].T stacks x's

    Then with a n x np coefficient matrix we obtain:
    Y_hat = B_hat * Z
    '''
    n = X.shape[0]
    T = X.shape[1]
    if T == 0:
        return np.array([]), np.array([])
    Y = X[:, p:]
    assert Y.shape[0] == n and Y.shape[1] == T - p, 'Issues with shape!'

    Z = np.zeros((n * p, T - p))
    for tau in range(p):
        Z[tau * n: (tau + 1) * n, :] = X[:, tau: tau - p]
    return Y, Z


def cross_validate(ZZT: np.array, YZT: np.array, X_test,
                   p: int, mu: float=0.1, tol=1e-6,
                   max_iter=100, warn_PSD=False, ret_B_err=False,
                   t_limit_sec=3600):
    '''
    Run through each possible combinations of parameters given by
    the lists lmbda, alpha, delta, and sigma and then fit the dwglasso
    model and cross validate it against the 1-step ahead prediction
    task on the data given in Z and Y.  Y_hat = B_hat * Z.
    '''
    t0 = time.time()

    def tlimit_func(*args, **kwargs):
        if time.time() - t0 >= t_limit_sec:
            return True
        return

    def f(x):
        l, a, d, s = x
        B_hat = dwglasso(ZZT=ZZT, YZT=YZT, p=p, lmbda=l, alpha=a,
                         mu=mu, delta=d, sigma=s, tol=tol,
                         warn_PSD=warn_PSD, ret_B_err=ret_B_err,
                         silent=True)
        Y_hat = np.dot(B_hat, Z)
        err = (np.linalg.norm(Y - Y_hat, ord='fro')**2) / T
        G = np.abs(sum([B_hat[:, tau * n:(tau + 1) * n]
                        for tau in range(p)]).T)
        G = G - np.diag(np.diag(G))
        G = G > 0  # The Granger-causality graph
        print('err = {:15.2f}'.format(err),
              '(l = %9.4f, a = %6.5f, d = %6.5f, s = %9.4f) ' % tuple(x),
              'Num edges: {:9d}'.format(np.sum(G)), end='\r')
        return err

    Y, Z = build_YZ(X_test, p)
    T = Y.shape[1]
    n = Y.shape[0]
    bounds = [(0, 2), (0, 1), (0, 1), (0, 100)]  # l a d s
    res = differential_evolution(f, bounds, disp=True, polish=False,
                                 maxiter=100, popsize=25,
                                 callback=tlimit_func, tol=1e-4)

    print()  # Newline
    print('Optimizer Success:', res.success)
    l, a, d, s = res.x
    print('Optimal parameters: lmbda = %0.5f, alpha = %0.5f, delta = %0.5f,'
          ' sigma = %0.5f' % (l, a, d, s))
    B_hat = dwglasso(ZZT=ZZT, YZT=YZT, p=p, lmbda=l, alpha=a,
                     mu=mu, delta=d, sigma=s, tol=tol,
                     warn_PSD=warn_PSD, ret_B_err=ret_B_err)
    print()
    plt.imshow(B_hat)
    plt.colorbar()
    plt.title('Optimal B_hat')
    plt.show()
    return B_hat


# This function is ridiculously complicated
def dwglasso(ZZT: np.array, YZT: np.array, p: int=1, lmbda: float=0.0,
             alpha: float=0.05, mu: float=0.1, delta=0, sigma=0,
             tol=1e-6, max_iter=100, warn_PSD=True, ret_B_err=False,
             silent=False, assert_params=True):
    '''Minimizes over B:

    1/(2T)||Y - BZ||_F^2 + lmbda[alpha||B||_F^2 + (1 - alpha)G_DW(B)

    via ADMM.  G_DW is the depth wise group regularizer

    \sum_{ij}||Bt_ij||_2 where Bt_ij is the p-vector (B(1)_ij ... B(p)_ij)
    e.g. the filter coefficients from xj -> xi.

    if lmbda = 0 we have simple ordinary least squares, and if alpha = 1
    then we have tikhonov regularization.  mu is a tuning parameter
    for ADMM convergence and unless lmbda = 0 or alpha = 1, we need mu > 0.
    '''
    if assert_params:
        assert alpha >= 0 and alpha <= 1, 'Required: alpha \in [0, 1]'
        assert lmbda >= 0, 'Required: lmbda >= 0'
        assert mu >= 0, 'Required: mu >= 0'  # 0 only if lmbda = 0 or alpha = 1
        assert sigma >= 0, 'Required: sigma >= 0'
        assert delta >= 0 and delta <= 1, 'Required: delta \in [0, 1]'

    # Proximity operators
    # @numba.jit(nopython=True, cache=True)
    def proxf_lu(V: np.array):
        '''
        proximity operator ||Y - BX||_F^2 + lmbda*(1 - alpha)||B||_F^2,
        implemented using an LU factorized covariance matrix.  This will
        work even if the covariance matrix is (due to numerical issues)
        not positive semi definite.
        '''
        return (lu_solve(lu_piv, YZT.T + V.T / mu,
                         overwrite_b=True, check_finite=False)).T

    def proxf_cho(V: np.array):
        '''
        proximity operator of ||Y - BX||_F^2 + lmbda*(1 - alpha)||B||_F^2,
        implemented using a cholesky factorized covariance matrix.  This
        requires the covariance matrix to be (numerically) positive
        semidefinite.
        '''
        return (cho_solve(L_and_lower, YZT.T + V.T / mu,
                          overwrite_b=True, check_finite=False)).T

    @numba.jit(nopython=True, cache=True)  # Dramatic speed up is achieved
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
                Vtij_l2 = 0
                for tau in range(p):  # Calling np.norm not valid w/ numba
                    Vtij_l2 += Vtij[tau]**2
                if Vtij_l2 == 0:
                    P[i, j::n] = 0
                else:
                    r = lmbda * (1 - alpha) * mu / Vtij_l2
                    P[i, j::n] = max(0, 1 - r) * Vtij
        return P

    def admm():
        def rel_err(Bxk, Bzk):  # F-norm difference between Bx and Bz per entry
            return (1 / (n * n * p)) * np.linalg.norm(Bxk - Bzk, 'f')**2

        # Init with 0s
        Bz, Bu = np.zeros((n, n * p)), np.zeros((n, n * p))
        Bx = proxf(Bz)
        k = 0
        rel_err_k = rel_err(Bx, Bz)
        while rel_err_k > tol and k < max_iter:  # ADMM iterations
            k += 1
            if not silent:
                print('iter:', k, '(1/pn^2)||Bx - Bz||_F^2 =',
                      rel_err_k, end='\r')
            sys.stdout.flush()
            Bx = proxf(Bz - Bu)
            Bz = proxg(Bx + Bu)
            Bu = Bu + Bx - Bz
            rel_err_k = rel_err(Bx, Bz)
        if not silent:
            print()  # Print out a newline
        if k >= max_iter:  # Should only ever reach k == max_iter
            if not silent:
                warnings.warn('Max iterations exceeded! rel_err = %e'
                              % rel_err_k, RuntimeWarning)
        return Bz

    # ----------------REAL FUNCTION ENTRY POINT-----------------------

    n = ZZT.shape[0] // p

    # Regularize the covariance estimate
    ZZT = (1 - delta) * ZZT + delta * np.diag(np.diag(ZZT)) +\
        sigma * np.eye(n * p)

    B_err = np.ones((n * p, n * p))
    if warn_PSD and not silent:  # Check if ZZT > 0
        try:
            cho_factor(ZZT)
        except np.linalg.LinAlgError as e:
            lmbda_min = eigh(ZZT, eigvals_only=True, turbo=False,
                             check_finite=False, eigvals=(0, 0))
            warnings.warn('ZZT is indefinite! lmbda_min(ZZT) = %e, err: %s'
                          % (lmbda_min, e.args))
            if ret_B_err:
                return B_err

    if lmbda == 0:  # OLS
        if not silent:
            print('OLS')
        try:
            L_and_lower = cho_factor(ZZT.T, overwrite_a=True)
            B = cho_solve(L_and_lower, YZT.T, overwrite_b=True,
                          check_finite=False).T
            return B
        except np.linalg.LinAlgError as e:  # ZZT is (probably) indefinite
            lu_piv = lu_factor(ZZT.T, overwrite_a=True)
            B = lu_solve(lu_piv, YZT.T, overwrite_b=True,
                         check_finite=False).T
            return B

    else:  # Regularized solution
        if alpha == 1:  # Tikhonov regularization with lmbda
            if not silent:
                print('L2 Regularization')
            # R and subsequently YZT are overwritten in the lu solver
            R = (ZZT + lmbda * np.eye(n * p))  # Has nothing to do with Rx
            try:
                L_and_lower = cho_factor(R.T, overwrite_a=True)
                B = cho_solve(L_and_lower, YZT.T, overwrite_b=True,
                              check_finite=False).T
                return B
            except np.linalg.LinAlgError as e:  # ZZT is (probably) indefinite
                lu_piv = lu_factor(R.T, overwrite_a=True)
                B = lu_solve(lu_piv, YZT.T, overwrite_b=True,
                             check_finite=False).T
                return B

        else:  # DWGLASSO
            if not silent:
                print('DWGLASSO')
            assert mu > 0, 'Required: mu > 0 (unless lmbda = 0, or alpha = 1)'
            if alpha == 0:
                if not silent:
                    warnings.warn('We need alpha > 0 to guarantee convergence,'
                                  ' to the optimal B matrix', RuntimeWarning)
            r = (1 + 2 * mu * lmbda * alpha) / mu
            R = (ZZT + r * np.eye(n * p))  # Has nothing to do with Rx
            try:
                L_and_lower = cho_factor(R.T, overwrite_a=True)
                proxf = proxf_cho
                return admm()
            except np.linalg.LinAlgError as e:  # ZZT is (probably) indefinite
                lu_piv = lu_factor(R.T, overwrite_a=True)
                proxf = proxf_lu
                return admm()


def main():
    p = 2
    assert p <= MAX_P and p >= 1, 'p must be in (1, MAX_P)!'

    ZZT = np.load(ZZT_FILE_PREFIX + str(p) + '_T' + '.npy')
    YZT = np.load(YZT_FILE_PREFIX + str(p) + '_T' + '.npy')
    XT = np.load(X_VALIDATE_FILE_PREFIX + '_T' + '.npy')

    Y_test, Z_test = build_YZ(XT, p)

    B_hat = dwglasso(ZZT, YZT, p, lmbda=370.0, alpha=0.2, tol=1e-11,
                     mu=0.1, max_iter=150, sigma=2.5, delta=0.1,
                     ret_B_err=False)
    print('Non 0 entries:', np.sum(np.abs(B_hat) > 0),
          '/', B_hat.size)
    plt.imshow(B_hat)
    plt.colorbar()
    plt.title('DWGLASSO Test Run on T')
    plt.show()

    # Verify, by rolling accross the station axis, that we have everything
    # correctly lined up.  We expect the lowest error on the non-rolled
    # Z_test matrix.  Then another dip after it's been rolled all the way back
    T = Y_test.shape[1]
    n = Y_test.shape[0]
    errs = []
    for i in range(n + 10):
        print('roll:', i, end='\r')
        Y_hat = np.dot(B_hat, np.roll(Z_test, i, axis=0))
        err_i = (np.linalg.norm(Y_hat - Y_test, ord='fro') ** 2) / T
        errs.append(err_i)
    plt.plot(range(n + 10), errs)
    plt.title('Verification of alignment')
    plt.xlabel('roll index')
    plt.ylabel('Error')
    plt.show()
    return


if __name__ == '__main__':
    main()
