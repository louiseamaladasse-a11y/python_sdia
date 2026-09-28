import numpy as np
import pytest

from functions import brownian_motion

def test_reproducibility():
    """Vérifie que la graine aléatoire garantit la même trajectoire."""
    rng1 = np.random.default_rng(seed=123)
    rng2 = np.random.default_rng(seed=123)

    w1, star1 = brownian_motion(50, [0.0, 0.0], 0.05, rng1)
    w2, star2 = brownian_motion(50, [0.0, 0.0], 0.05, rng2)

    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(star1, star2)


def test_boundary_intersection_norm():
    """Vérifie que ||W*|| vaut exactement 1 quand la frontière est franchie."""
    rng = np.random.default_rng(seed=42)
    # Un grand pas garantit la sortie du cercle
    trajectory, w_star = brownian_motion(100, [0.5, 0.5], 0.1, rng)

    assert w_star is not None
    # On vérifie que la norme vaut 1 à 1e-6 près
    assert np.isclose(np.linalg.norm(w_star), 1.0, atol=1e-6)


def test_no_boundary_crossing():
    """Vérifie que w_star est None si la particule reste dans le cercle."""
    rng = np.random.default_rng(seed=42)
    # 2 itérations très courtes au centre ne sortent pas du cercle
    trajectory, w_star = brownian_motion(2, [0.0, 0.0], 1e-5, rng)

    assert w_star is None
    assert len(trajectory) == 3  # N + 1 points


def test_initial_point_and_shape():
    """Vérifie le point initial et la forme de la trajectoire."""
    rng = np.random.default_rng(seed=0)
    x0 = [0.2, -0.3]
    trajectory, _ = brownian_motion(10, x0, 0.01, rng)

    # Le premier point doit être le point de départ
    np.testing.assert_allclose(trajectory[0], x0)
    # La deuxième dimension doit être 2 (2D)
    assert trajectory.shape[1] == 2
    # La longueur doit être au maximum N + 1
    assert len(trajectory) <= 11


def test_input_formats():
    """Vérifie que la fonction accepte aussi bien des listes que des ndarrays."""
    rng1 = np.random.default_rng(seed=1)
    rng2 = np.random.default_rng(seed=1)

    w_list, _ = brownian_motion(10, [0.1, 0.1], 0.01, rng1)
    w_arr, _ = brownian_motion(10, np.array([0.1, 0.1]), 0.01, rng2)

    np.testing.assert_array_equal(w_list, w_arr)
