import numpy as np
cimport numpy as np
import bottleneck

def k_nn(x, x_train, class_train, k):

    # On commence par calculer la norme euclidienne entre le point x et tous les autres points de x_train
    distances = np.linalg.norm(x_train - x, axis=1)

    # On stocke les indices des distances triées dans l'ordre croissant par argsort
    #indices_distances = np.argsort(distances)

    indices_distances = bottleneck.argpartition(distances, kth=k-1)

    # On ne conserve que les k premiers indices qui vont correspondre aux indices des k plus proches voisins
    indices_k_nn = indices_distances[:k]

    # bincount retourne le nombre d'occurences par classe des k plus proches voisins
    counts = np.bincount(class_train[indices_k_nn].astype(int))

    # On retrouve la classe majoritaire en récupérant l'indice avec le plus d'occurences
    classe_x = np.argmax(counts)

    return classe_x


def k_nn_v1(x, x_train, class_train, k):
    n_samples = x_train.shape[0]
    n_features = x_train.shape[1]

    # Tableau numpy pour stocker les distances
    distances = np.zeros(n_samples)

    # Double boucle for
    for i in range(n_samples):
        s = 0.0
        for j in range(n_features):
            diff = x_train[i, j] - x[j]
            s += diff * diff
        distances[i] = np.sqrt(s)

    indices_distances = np.argsort(distances)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn].astype(int))
    return np.argmax(counts)


def k_nn_v2(double[:] x, double[:, :] x_train, long[:] class_train, int k):
    cdef int n_samples = x_train.shape[0]
    cdef int n_features = x_train.shape[1]
    cdef int i, j
    cdef double s, diff

    # Création du tableau numpy et récupération de sa memoryview
    distances_np = np.zeros(n_samples, dtype=np.float64)
    cdef double[:] distances = distances_np

    for i in range(n_samples):
        s = 0.0
        for j in range(n_features):
            diff = x_train[i, j] - x[j]
            s += diff * diff
        distances[i] = s ** 0.5  # sqrt en C

    indices_distances = np.argsort(distances_np)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn])
    return np.argmax(counts)

@cython.boundscheck(False)
@cython.wraparound(False)

def k_nn_v3(double[:] x, double[:, :] x_train, long[:] class_train, int k):
    cdef int n_samples = x_train.shape[0]
    cdef int n_features = x_train.shape[1]
    cdef int i, j
    cdef double s, diff

    # Création du tableau numpy et récupération de sa memoryview
    distances_np = np.zeros(n_samples, dtype=np.float64)
    cdef double[:] distances = distances_np

    for i in range(n_samples):
        s = 0.0
        for j in range(n_features):
            diff = x_train[i, j] - x[j]
            s += diff * diff
        distances[i] = s ** 0.5  # sqrt en C

    indices_distances = np.argsort(distances_np)
    indices_k_nn = indices_distances[:k]
    counts = np.bincount(class_train[indices_k_nn])
    return np.argmax(counts)
