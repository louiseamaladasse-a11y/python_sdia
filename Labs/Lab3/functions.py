import numpy as np

def brownian_motion(niter, x, step, rng):
    # On convertit en ndarray pour qu'on compare des objets du même type
    x = np.asarray(x, dtype=float)

    # On vérifie que le point de départ est dans la boucle
    if np.dot(x, x) > 1.0:
        raise ValueError("Le point x doit être à l'intérieur de la boule unité.")


    w = np.zeros((niter + 1, 2))
    w[0] = x

    G = rng.normal(0, 1, size=(niter, 2))

    i = 1
    w_star = None

    sqrt_step = np.sqrt(step)

    for i in range(1, niter + 1):
        # Calcul du point suivant
        w[i] = w[i - 1] + sqrt_step * G[i - 1]

        # Test de sortie du cercle
        if np.dot(w[i], w[i]) > 1.0:
            w_prev = w[i - 1]
            d = w[i] - w_prev

            # Coefficients du polynôme a*t^2 + b*t + c = 0
            a = np.dot(d, d)
            b = 2.0 * np.dot(w_prev, d)
            c = np.dot(w_prev, w_prev) - 1.0

            # Calcul des racines avec np.roots
            roots = np.roots([a, b, c])
            t_star = [t for t in roots if 0 <= t <= 1][0]

            # Calcul du point d'intersection W*
            w_star = w_prev + t_star * d

            # Tronquer la trajectoire aux pas effectués
            w = w[:i + 1]
            break

    return w, w_star
