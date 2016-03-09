import numpy as np
from choldate import choldowndate, cholupdate
from numpy.linalg import cholesky


class Helper(object):

    def chol_update(self, L, X):
        """
        Cholesky Rank 1 Update
        This function computes the lower triangular cholesky decomposition L' of matrix A' from L (the cholesky decomp of A)
        where: A' = A + x*x^T.
        Based on the pseudocode in the wiki page https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update

        :param L: Lower triange matrix from Cholesky Decomposition
        :param X: Word Vector with same column dimensionality as L
        :return: updated lower triangle matrix
        """

        assert L.shape[1] == X.shape[0], "cholesky lower triangle matrix dim != word vec dim"
        # cholupdate(L.T, X)

        for k in range(X.shape[0]):
            r = np.sqrt(L[k, k]**2 + X[k]**2)
            c = r / L[k, k]
            s = X[k] / L[k, k]
            L[k, k] = r

            for i in range(k+1, X.shape[0]):
                L[i, k] = (L[i, k] + (s * X[i])) / c
                X[i] = (c * X[i]) - (s * L[i, k])

        return L.T


    def chol_downdate(self, L, X):
        """
        Cholesky Rank 1 Update
        :param L: Lower triangle matrix from Cholesky Decomposition
        :param X: Word Vector with same column dimensionality as L
        :return: updated lower triangle matrix
        """
        assert L.shape[1] == X.shape[0]
        # choldowndate(L.T, X)

        for k in range(X.shape[0]):
            r = np.sqrt(L[k, k]**2 - X[k]**2)
            c = r / L[k, k]
            s = X[k] / L[k, k]
            L[k, k] = r

            for i in range(k+1, X.shape[0]):
                L[i, k] = (L[i, k] - (s * X[i])) / c
                X[i] = (c * X[i]) - (s * L[i, k])
        if np.isnan(r):
            print "YOU GOT NANAANANA"
        return L.T