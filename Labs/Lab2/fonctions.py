import numpy as np



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



def gradient2D_adjoint(Y):

    """
    Compute the adjoint of the 2D discrete gradient operator.

    Parameters
    ----------
    Y : array_like
        Input array of shape (2, M, N), where Y[0] is the horizontal
        component and Y[1] is the vertical component.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (M, N) corresponding to D*(Y), the adjoint
        of the 2D discrete gradient operator.
    """
    dim, m, n = np.shape(Y)



    # On sépare les deux matrices en entrée
    Y_h = Y[0]
    Y_v = Y[1]

    # Horizontal
    D_h = -np.diff(
        np.concatenate(
            (np.zeros((m, 1)), Y_h[:, :-1], np.zeros((m, 1))),
            # On reconstruit une matrice de la forme [0, y_1, ..., y_(n-1), 0] qui nous permet d'utiliser np.diff
            # il suffit de multiplier par -1 pour obtenir le résultat voulu
            axis=1
        ),
        axis=1
    )

    # Vertical
    D_v = -np.diff(
        np.concatenate(
            (np.zeros((1, n)), Y_v[:-1, :], np.zeros((1, n))),
            axis=0
        ),
        axis=0
    )

    return D_h + D_v
