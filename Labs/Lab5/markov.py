import numpy as np

def markov(rho, A, nmax, rng) :
    # On convertit tout dans le même format
    rho = np.asarray(rho)
    A = np.asarray(A)
    N = len(rho)

    assert A.shape == (N,N) # On vérifie que A est de taille (N,N), carrée
    # On vérifie que toutes les probabilités sont positives
    assert np.all(rho>=0)
    assert np.all(A >= 0)
    assert np.isclose(np.sum(rho), 1.0) # La somme des éléments de rho vaut 1
    assert np.allclose(np.sum(A, axis=1), 1.0) # La matrice 1 est stochastique, la somme de chaque ligne vaut 1

    X = np.zeros(nmax + 1, dtype=int)
    states = np.arange(N)

    X[0] = rng.choice(states, p=rho)

    for q in range(nmax):
        X[q + 1] = rng.choice(N, p=A[X[q]])

    return X
