import numpy as np
from numba import jit

def gradient2D(X):

    """
    Compute the 2D discrete gradient of a matrix.

    Parameters
    ----------
    X : array_like
        Input matrix of shape (M, N).

    Returns
    -------
    numpy.ndarray
        Gradient array of shape (2, M, N), where the first component
        contains the horizontal differences and the second component
        contains the vertical differences.

    Raises
    ------
    AssertionError
        If X has more than two dimensions.
    """
    X = np.asarray(X)

    if X.ndim > 2:
        raise AssertionError("erreur de dimension")

    m, n = X.shape

    # X D_h : différences entre colonnes
    grad_h = np.concatenate(
        (np.diff(X, axis=1), np.zeros((m, 1))),
        axis=1
    )

    # D_v X : différences entre lignes
    grad_v = np.concatenate(
        (np.diff(X, axis=0), np.zeros((1, n))),
        axis=0
    )

    return np.array([grad_h, grad_v])






def tv(X):
    """
    Compute the discrete isotropic total variation of a matrix.

    Parameters
    ----------
    X : array_like
        Input matrix of shape (M, N).

    Returns
    -------
    float
        Total variation of X.
    """

    G = gradient2D(X)

    return np.sum(np.sqrt(G[0]**2 + G[1]**2))




@jit(nopython=True)
# Grâce à cette ligne, les boucles seront exécutées plus efficacement avec Numba


def tv_numba(X):
    """
    Compute the discrete isotropic total variation of a matrix
    using Numba acceleration.
    """

    m, n = X.shape
    total = 0.0

    for i in range(m):
        for j in range(n):

            # gradient horizontal
            if j < n - 1:
                grad_h = X[i, j + 1] - X[i, j]
            else:
                grad_h = 0.0

            # gradient vertical
            if i < m - 1:
                grad_v = X[i + 1, j] - X[i, j]
            else:
                grad_v = 0.0

            total += np.sqrt(grad_h**2 + grad_v**2)

    return total
