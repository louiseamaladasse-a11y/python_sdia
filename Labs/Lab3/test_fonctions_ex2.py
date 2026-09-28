import numpy as np
import pytest

from fonctions_ex2 import ideal_lowpass_filter

def test_dimensions_and_type():
    """
    Vérifie que la fonction renvoie bien une matrice de la même dimension 
    que l'entrée, et qu'elle ne contient que des réels (pas de complexes).
    """
    X = np.random.rand(100, 100)
    X_filtered = ideal_lowpass_filter(X, 10, 10)
    
    assert X_filtered.shape == X.shape
    assert not np.iscomplexobj(X_filtered)

def test_all_pass():
    """
    Vérifie qu'un filtre englobant toutes les fréquences ne modifie pas l'image.
    Si les fréquences de coupure dépassent la taille de l'image, le masque 
    ne contient que des 1, l'image doit donc être restituée à l'identique.
    """
    X = np.random.rand(64, 64)
    # Fréquences de coupure volontairement très larges
    X_filtered = ideal_lowpass_filter(X, 100, 100)
    
    assert np.allclose(X, X_filtered)

def test_zero_pass():
    """
    Vérifie que si on applique des fréquences de coupure de 0, le masque
    est vide, tout est multiplié par 0, et l'image retournée est entièrement noire.
    """
    X = np.random.rand(50, 50)
    X_filtered = ideal_lowpass_filter(X, 0, 0)
    
    # L'image filtrée ne doit contenir que des zéros
    assert np.allclose(X_filtered, np.zeros_like(X))

def test_rectangular_image():
    """
    Vérifie que la fonction gère correctement les images non carrées 
    (dimensions M et N différentes).
    """
    X = np.random.rand(40, 80)
    X_filtered = ideal_lowpass_filter(X, 10, 20)
    
    assert X_filtered.shape == (40, 80)