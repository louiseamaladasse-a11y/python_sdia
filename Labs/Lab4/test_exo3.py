import numpy as np

from fonctions_exo3 import tv, tv_numba


def test_tv_numba_same_result():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    resultat_numpy = tv(X)
    resultat_numba = tv_numba(X)

    assert np.isclose(resultat_numpy, resultat_numba)


def test_tv_numba_random_matrix():
    np.random.seed(0)
    X = np.random.rand(10, 10)

    resultat_numpy = tv(X)
    resultat_numba = tv_numba(X)

    assert np.isclose(resultat_numpy, resultat_numba)
