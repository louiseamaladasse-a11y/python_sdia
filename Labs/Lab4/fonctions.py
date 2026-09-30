import numpy as np

def k_nn(x, x_train, class_train, k):

    """
    Classifie un point x avec l'algorithme des k plus proches voisins.

    Parameters
    ----------
    x : array
        Point à classifier.
    x_train : array
        Données d'entraînement.
    class_train : array
        Classes associées aux données d'entraînement.
    k : int
        Nombre de voisins considérés.

    Returns
    -------
    int
        Classe prédite pour x.
    """

    # On commence par calculer la norme euclidienne entre le point x et tous les autres points de x_train
    distances = np.linalg.norm(x_train - x, axis=1)

    # On stocke les indices des distances triées dans l'ordre croissant par argsort
    indices_distances = np.argsort(distances)

    # On ne conserve que les k premiers indices qui vont correspondre aux indices des k plus proches voisins
    indices_k_nn = indices_distances[:k]

    # bincount retourne le nombre d'occurences par classe des k plus proches voisins
    counts = np.bincount(class_train[indices_k_nn].astype(int))

    # On retrouve la classe majoritaire en récupérant l'indice avec le plus d'occurences
    classe_x = np.argmax(counts)

    return classe_x
