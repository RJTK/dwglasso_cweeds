'''
A 'helper' script to calculate and subsequently cache the
covariance matrices ZZT and YZT.  This is time consuming so it's
certainly wise to cache this caculation.  This is basically a prereq
to running the dwglasso algorithm.

NOTE: This file is intended to be executed by make from the top
level of the project directory hierarchy.  We rely on os.getcwd()
and it will not work if run directly as a script from this directory.
'''
import sys
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement, repeat, starmap
from src.conf import ZZT_FILE_PREFIX, YZT_FILE_PREFIX, HDF_FINAL_FILE,\
    LOCATIONS_KEY, MAX_P, TEMPERATURE_TS_ROOT


def periodogram_covar(x: np.array, y: np.array, tau: int, p: int):
    '''Takes in numpy arrays x and y, an int value tau for the lag and
    another int p for the maximum lag and returns the periodogram
    estimate of the covariance.
    '''
    assert np.allclose(x.mean(), 0), 'Signal x must be 0 mean!'
    assert np.allclose(y.mean(), 0), 'Signal y must be 0 mean!'
    T = len(x) - p
    if tau == 0:
        return (1 / T) * np.dot(x[p:], y[p:])
    else:
        return (1 / T) * np.dot(x[p:], y[p - tau:-tau])


# Exists essentially just to help implement the convenient notation in
# the function periodogram_covar_matrices.
class ColSelector(object):
    '''
    A helper object to select out the temperature data from our hdf store.
    '''
    def __init__(self, hdf: pd.io.pytables.HDFStore, keys, column: str):
        '''Takes in an HDFStore D, an iterable of keys, and the column we
        want to select from each of the key locations.  Precisely,
        each key location (hdf[keys[i]]) should contain a pd.DataFrame
        object D having the given column: D[column].

        If we have t = ColSelector(hdf, keys, column) then t.keys()
        will simply return keys, and t[k] will return
        hdf[k][column]
        '''
        self._keys = keys
        self.column = column
        self.hdf = hdf
        self.shape = (len(self.hdf[self._keys[0]]), len(self._keys))
        return

    def __getitem__(self, k):
        '''selector[k]'''
        return self.hdf[k][self.column]

    def keys(self):
        return self._keys


# Use the ColSelector class to give convenient access to an HDFStore
# e.g. give ColSelector the keys we want to iterate through, and the
# column we want to access.
def periodogram_covar_matrices(D, p: int):
    '''Makes use of the helper functions periodogram_covar to calculate
    the covariance matrices Rx(0) ... Rx(p) where x_i is the i'th
    column of D.  The matrices Rx(0) ... Rx(p - 1) form the top row
    of ZZT and Rx(1) ... Rx(p) form YZT.

    We will return the raw estimates of covariances.  For large
    systems, these estimates are unlikely to be positive semidefinite.
    Shrinkage and regularization should be added later in the
    pipeline.
    '''
    n = D.shape[1]
    Rx = np.zeros((n, n * (p + 1)))

    # I'm unsure about the ordering conventions for D.keys()
    # when called on a pd.DataFrame.  But, in this case I'm only
    # using it with my own ColSelector class, where I know that
    # D.keys() is a genuine, ordered list.  This ordering is of
    # critical importance for later plotting as I'm using no other
    # way to keep track of which covariance corresponds to which
    # weather station.
    for ixi, jxj in combinations_with_replacement(
            enumerate(D.keys()), 2):
        i, xi = ixi
        j, xj = jxj
        print('p = %d, Cov(x%d, x%d)' % (p, i, j), end='\r')
        sys.stdout.flush()
        xi = D[xi].values  # D[xi] should be a pd.Series.
        xj = D[xj].values
        Rx[i, j::n] = np.fromiter(starmap(periodogram_covar,
                                          zip(repeat(xi, p + 1),
                                              repeat(xj, p + 1),
                                              range(p + 1),
                                              repeat(p, p + 1))),
                                  float, count=p + 1)
        # Fill in the rest by symmetry
        Rx[j, i::n] = Rx[i, j::n]
    print()
    Rx = list(np.split(Rx, p + 1, axis=1))
    return Rx


def form_ZZT(Rx, delta=0.01):
    '''Forms the matrix ZZT from the list of Rx matrices
    Rx = [Rx(0) Rx(1) ... Rx(p)] (n x n*(p + 1))

    ZZT is a np x np block toeplitz form from the 0 to p - 1 lagged
    covariance matrices of the n-vector x(t).

    CARE: The matrix returned from this function is unlikely to be
    positive semidefinite for large n.  The shrinkage and other
    manipulation necessary to ensure ZZT > 0 should be handled later
    in the pipeline and the parameters involved should be considered
    as true parameters of the model.
    '''
    p = len(Rx) - 1
    n = Rx[0].shape[0]
    ZZT = np.zeros((n * p, n * p))

    for i in range(p):
        for j in range(p):
            ZZT[i * n:(i + 1) * n, j * n:(j + 1) * n] = Rx[abs(i - j)]

    return ZZT


def form_YZT(Rx):
    '''Forms the matrix YZT from the list of Rx matrices
    Rx = [Rx(0) Rx(1) ... Rx(p)] (n x n*(p + 1))

    YZT is an n x np matrix [Rx(1) ... Rx(p)]
    '''
    YZT = np.hstack(Rx[1:])  # [Rx(1) ... Rx(p)]
    return YZT


def main():
    hdf_final = pd.HDFStore(HDF_FINAL_FILE, mode='a')
    wbans = hdf_final[LOCATIONS_KEY]['WBAN']

    # The ORDER of this array is of CRITICAL importance for later plotting
    keys = [TEMPERATURE_TS_ROOT + '/wban_' + wban + '/D'
            for wban in wbans]
    col = 'dT'
    D = ColSelector(hdf_final, keys, col)
    for p in range(1, MAX_P + 1):
        Rxp = periodogram_covar_matrices(D, p)
        ZZTp = form_ZZT(Rxp)
        YZTp = form_YZT(Rxp)
        np.save(ZZT_FILE_PREFIX + str(p) + '_dT', ZZTp)
        np.save(YZT_FILE_PREFIX + str(p) + '_dT', YZTp)

    col = 'T'
    D = ColSelector(hdf_final, keys, col)
    for p in range(1, MAX_P + 1):
        Rxp = periodogram_covar_matrices(D, p)
        ZZTp = form_ZZT(Rxp)
        YZTp = form_YZT(Rxp)
        np.save(ZZT_FILE_PREFIX + str(p) + '_T', ZZTp)
        np.save(YZT_FILE_PREFIX + str(p) + '_T', YZTp)
    return


if __name__ == '__main__':
    main()
