import numpy as np
from choldate import choldowndate, cholupdate
from numpy.linalg import cholesky
from numba import jit

class Helper(object):
    @jit(cache=True)
    def chol_update(self, L, X):
        """
        Cholesky Rank 1 Update
        This function computes the lower triangular cholesky decomposition L' of matrix A' from L (the cholesky decomp of A)
        where: A' = A + x*x^T.
        Based on the pseudocode in the wiki page https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update

        :param L_c: Lower triange matrix from Cholesky Decomposition
        :param X: Word Vector with same column dimensionality as L
        :return: updated lower triangle matrix
        """

        # assert L.shape[1] == X.shape[0], "cholesky lower triangle matrix dim != word vec dim"
        # cholupdate(L.T, X)
        L_c = np.copy(L)
        for k in range(X.shape[0]):
            r = np.sqrt(L_c[k, k] ** 2 + X[k] ** 2)
            c = r / L_c[k, k]
            s = X[k] / L_c[k, k]
            L_c[k, k] = r

            for i in range(k+1, X.shape[0]):
                L_c[i, k] = (L_c[i, k] + (s * X[i])) / c
                X[i] = (c * X[i]) - (s * L_c[i, k])
        if np.isnan(r) or np.isinf(r):
            print "Your updater sucks, you have nans or infs"; return L
        return L_c

    @jit(cache=True)
    def chol_downdate(self, L, X):
        """
        Cholesky Rank 1 Update
        :param L: Lower triangle matrix from Cholesky Decomposition
        :param X: Word Vector with same column dimensionality as L
        :return: updated lower triangle matrix
        """
        # assert L.shape[1] == X.shape[0]
        # choldowndate(L.T, X)
        L_c = np.copy(L) # in-place computations are faster
        for k in range(X.shape[0]):
            r = np.sqrt(L_c[k, k]**2 - X[k]**2)
            c = r / L_c[k, k]
            s = X[k] / L_c[k, k]
            L_c[k, k] = r

            for i in range(k+1, X.shape[0]):
                L_c[i, k] = (L_c[i, k] - (s * X[i])) / c
                X[i] = (c * X[i]) - (s * L_c[i, k])
        if np.isnan(r) or np.isinf(r):
            print "YOU GOT Nans or infs: learn to code better shmuck"
            return L # good reason for making copy - return if downdate becomes unstable
        return L_c