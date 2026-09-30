import numpy as np
from fonctions import k_nn


def test_knn_class_1():
    x_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])

    class_train = np.array([1, 1, 1, 2, 2, 2])

    x = np.array([0.2, 0.2])

    prediction = k_nn(x, x_train, class_train, k=3)

    assert prediction == 1


def test_knn_class_2():
    x_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ])

    class_train = np.array([1, 1, 1, 2, 2, 2])

    x = np.array([5.2, 5.1])

    prediction = k_nn(x, x_train, class_train, k=3)

    assert prediction == 2