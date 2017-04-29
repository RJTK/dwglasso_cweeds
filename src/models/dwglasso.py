'''Implements DWGLASSO and associated helper functions'''
import numba
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.linalg import lu_factor, lu_solve

def dwglasso():
    '''
    '''

    @numba.jit(nopython = False, cache = True)
    def proxg(V):
        '''proximal operator of alpha*lmbda*sum_ij||B_ij||_2 See DWGLASSO
        paper for details
        '''
        n = V.shape[0]
        p = V.shape[1] / n
        P = np.empty((n, n*p))
        for i in range(n):
            for j in range(n):
                Vtij = V[i, j::n]
                Vtij_l2 = np.linalg.norm(Vtij, ord = 2)
                if Vtij_l2 == 0:
                    P[i, j::n] = 0
                else:
                    r = lmbda*(1 - alpha)*mu / Vtij_l2
                    P[i, j::n] = max(0, 1 - r)*Vtij
        return P
                
                    
                
    return


