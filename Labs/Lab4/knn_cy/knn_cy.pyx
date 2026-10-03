import numpy as np
cimport cython
import bottleneck
from libc.math cimport sqrt

# STEP-BY-STEP OPTIMIZATION

DTYPE = np.float64


# Fonction knn telle quelle pour comparer
def k_nn_v0(x, x_train, class_train, k):
    distances = np.linalg.norm(x_train - x, axis=1)
    indices_distances = bottleneck.argpartition(distances, kth=k-1)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn].astype(int))
    return np.argmax(counts)


# Comme conseillé et constaté, la fonction np.linalg.norm consomme beaucoup donc on l'améliore en la découpant en boucle et en ajoutant les types cython, x et x_train restent des objets numpy pour l'instant
def k_nn_v1(x, x_train, class_train, int k):
    assert x.dtype == DTYPE
    assert x_train.dtype == DTYPE

    cdef Py_ssize_t n = x_train.shape[0]
    cdef Py_ssize_t d = x_train.shape[1]
    cdef Py_ssize_t i, j
    cdef double diff, somme

    distances = np.zeros(n, dtype=DTYPE)

    for i in range(n):
        somme = 0.0
        for j in range(d):
            diff = x_train[i, j] - x[j]
            somme += diff * diff
        distances[i] = sqrt(somme)

    # Le reste est conservé en numpy
    indices_distances = bottleneck.argpartition(distances, kth=k-1)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn].astype(int))
    return np.argmax(counts)


# On remplace les tableaux numpy en memoryview pour optimiser
def k_nn_v2(double[:] x, double[:, :] x_train, class_train, int k):
    cdef Py_ssize_t n = x_train.shape[0]
    cdef Py_ssize_t d = x_train.shape[1]
    cdef Py_ssize_t i, j
    cdef double diff, somme

    distances = np.zeros(n, dtype=DTYPE)
    cdef double[:] distances_view = distances   # même mémoire, pas de copie

    for i in range(n):
        somme = 0.0
        for j in range(d):
            diff = x_train[i, j] - x[j]
            somme += diff * diff
        distances_view[i] = sqrt(somme)

    # On écrit via la vue, on passe le vrai tableau numpy à bottleneck
    indices_distances = bottleneck.argpartition(distances, kth=k-1)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn].astype(int))
    return np.argmax(counts)


#  Version finale : comme indiqué, on désactive les sécurités par défaut Cython pour gagner du temps :
@cython.boundscheck(False)
@cython.wraparound(False)
def k_nn_v3(double[:] x, double[:, :] x_train, class_train, int k):
    cdef Py_ssize_t n = x_train.shape[0]
    cdef Py_ssize_t d = x_train.shape[1]
    cdef Py_ssize_t i, j
    cdef double diff, somme

    distances = np.zeros(n, dtype=DTYPE)
    cdef double[:] distances_view = distances

    for i in range(n):
        somme = 0.0
        for j in range(d):
            diff = x_train[i, j] - x[j]
            somme += diff * diff
        distances_view[i] = sqrt(somme)

    indices_distances = bottleneck.argpartition(distances, kth=k-1)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn].astype(int))
    return np.argmax(counts)


# Comme conseillé dans le tutoriel, on déclare le tableau Numpy comme contingu pour avoir des gains supplémentaires
@cython.boundscheck(False)
@cython.wraparound(False)
def k_nn_v4(double[::1] x, double[:, ::1] x_train, class_train, int k):
    cdef Py_ssize_t n = x_train.shape[0]
    cdef Py_ssize_t d = x_train.shape[1]
    cdef Py_ssize_t i, j
    cdef double diff, somme

    distances = np.zeros(n, dtype=DTYPE)
    cdef double[::1] distances_view = distances

    for i in range(n):
        somme = 0.0
        for j in range(d):
            diff = x_train[i, j] - x[j]
            somme += diff * diff
        distances_view[i] = sqrt(somme)

    indices_distances = bottleneck.argpartition(distances, kth=k-1)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn].astype(int))
    return np.argmax(counts)
