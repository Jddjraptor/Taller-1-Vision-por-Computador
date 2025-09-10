import numpy as np

def distorsion_radial(puntos, k1, k2):
     
    # Separar array
    puntos = np.asarray(puntos)
    x = puntos[:, 0]
    y = puntos[:, 1]
    
    # Calcular r2
    r2 = x**2 + y**2
    
    # Calcular factor de distorsión
    factor = 1 + k1*r2 + k2*(r2**2)
    
    # Calcular x_corr, y_corr
    x_corr = x * factor
    y_corr = y * factor
    
    return np.vstack([x_corr, y_corr]).T

def longitud_focal(puntos_3D, f):
    
    # Separar array
    puntos_3D = np.asarray(puntos_3D)
    X = puntos_3D[:, 0]
    Y = puntos_3D[:, 1]
    Z = puntos_3D[:, 2]
    
    # Proyección pinhole
    x_proj = f * (X / Z)
    y_proj = f * (Y / Z)
    
    return np.vstack([x_proj, y_proj]).T
