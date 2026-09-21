import numpy as np
import pytest

from fonctions import gradient2D, tv, gradient2D_adjoint


def test_square_constant_matrix():
    X = np.ones((3, 3))
    G = gradient2D(X)

    assert G.shape == (2, 3, 3) # On vérifie les dimensions
    assert np.all(G == 0) # On vérifie que le gradient est nul


def test_non_square_constant_matrix():
    X = np.ones((2, 4))
    G = gradient2D(X)

    # Idem pour la matrice non carrée
    assert G.shape == (2, 2, 4)
    assert np.all(G == 0)


def test_wrong_dimension():
    X = np.zeros((2, 2, 2))
    # On vérifie que l'assertion se déclenche bien
    with pytest.raises(AssertionError):
        gradient2D(X)






# Test TV

def test_tv_constant_matrix():
    X = np.ones((3, 3))
    # Comme le gradient d'une matrice constante est nul, on vérifie que la variation totale soit nulle
    assert tv(X) == 0



def test_gradient2D_adjoint():

    rng = np.random.default_rng(0)

    m, n = 3, 4

    X = rng.random((m, n))
    Y = rng.random((2, m, n))

    # On vérifie la taille
    DstarY = gradient2D_adjoint(Y)
    assert DstarY.shape == (m, n)

    # On vérifie l'égalité des produits scalaires qui caractérise l'adjoint
    left = np.vdot(gradient2D(X), Y)
    right = np.vdot(X, DstarY)

    assert np.allclose(left, right)
    # On utilise allclose ici pour permettre la comparaison de flottants sans erreur d'arrondi
