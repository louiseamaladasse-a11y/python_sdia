import numpy as np

def brownian_motion(niter, x, step, rng):
    # On convertit en ndarray pour qu'on compare des objets du même type
    x = np.asarray(x, dtype=float)

    # On vérifie que le point de départ est dans la boule
    if np.dot(x, x) > 1.0:
        raise ValueError("Le point x doit être à l'intérieur de la boule unité.")


    w = np.zeros((niter + 1, 2))
    w[0] = x

    # On génère les pas gaussiens pour chaque itération
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

            # On utilise l'indice : le segment entre Wn et Wn-1 est la droite w(t) = Wn-1 + t*d
            # On cherche w(t) tel que ||w(t)||^2 = 1
            # Donc ||Wn-1 + t*d||^2 = 1
            # Soit ||Wn-1||^2 + t*2*<Wn-1,d> + t^2*||d||^2 - 1 = 0
            # On cherche donc les racines de ce polynôme

            # Coefficients du polynôme a*t^2 + b*t + c = 0
            a = np.dot(d, d)
            b = 2.0 * np.dot(w_prev, d)
            c = np.dot(w_prev, w_prev) - 1.0

            # Calcul des racines avec np.roots
            roots = np.roots([a, b, c])
            # Comme c est négatif (Wn-1 est dans la boule B(0,1) ), on a une racine positive et une négative, on cherche la racine positive (t€[0,1])
            t_star = [t for t in roots if 0 <= t <= 1][0]

            # Calcul du point d'intersection W*
            w_star = w_prev + t_star * d

            # Tronquer la trajectoire aux pas effectués
            w = w[:i + 1]
            break

    return w, w_star


def ideal_lowpass_filter(X, fcy, fcx):
    """
    Filtre l'image X avec un filtre passe-bas idéal.
    fcy, fcx : nombre d'échantillons à conserver de part et d'autre du centre
    pour les axes y (vertical) et x (horizontal).
    """
    # 1. Récupération des dimensions de l'image
    M, N = X.shape

    # 2. Transformée de Fourier et centrage (comme vu à la question 2)
    F_X = np.fft.fft2(X)
    F_X_shifted = np.fft.fftshift(F_X)

    # 3. Création du masque (filtre idéal)
    # On initialise une matrice de zéros de la même taille que l'image
    mask = np.zeros((M, N))

    # On trouve les coordonnées du centre de l'image
    center_y, center_x = M // 2, N // 2

    # On définit les limites de la fenêtre à conserver en évitant de déborder
    y_min = max(0, center_y - fcy)
    y_max = min(M, center_y + fcy)
    x_min = max(0, center_x - fcx)
    x_max = min(N, center_x + fcx)

    # On place des 1 dans le rectangle central défini par les fréquences de coupure
    mask[y_min:y_max, x_min:x_max] = 1

    # 4. Application du filtre
    # Les fréquences centrales sont multipliées par 1 (conservées)
    # Les autres sont multipliées par 0 (annulées)
    F_X_filtered_shifted = F_X_shifted * mask

    # 5. Retour au domaine spatial (image visible)
    # On annule le décalage (shift) avant de faire la transformée inverse
    F_X_filtered = np.fft.ifftshift(F_X_filtered_shifted)

    # On calcule la transformée de Fourier inverse
    X_filtered = np.fft.ifft2(F_X_filtered)

    # Le résultat d'une FFT inverse est complexe, on ne garde que la partie réelle
    # qui correspond aux valeurs d'intensité des pixels de l'image filtrée
    return np.real(X_filtered)
