import numpy as np

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