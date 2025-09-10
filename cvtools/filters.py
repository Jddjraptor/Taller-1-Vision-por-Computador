import numpy as np
from PIL import Image

def convolucion_generica(imagen, kernel):

    # Asegurar float para evitar overflow
    imagen = imagen.astype("float32")
    kernel = np.flipud(np.fliplr(kernel))
    
    # Dimensiones
    h, w = imagen.shape[:2]
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Padding de la imagen
    if imagen.ndim == 2:
        padded = np.pad(imagen, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        salida = np.zeros_like(imagen)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kh, j:j+kw]
                salida[i, j] = np.sum(region * kernel)
    else:
        salida = np.zeros_like(imagen)
        for c in range(3):
            padded = np.pad(imagen[:,:,c], ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    salida[i, j, c] = np.sum(region * kernel)

    return salida

def filtro_Sobel_X(imagen):

    # Aplicar filtro Sobel en X
    kernel_sobel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
                               
    return convolucion_generica(imagen, kernel_sobel_x)

def filtro_Sobel_Y(imagen):

    # Aplicar filtro Sobel en Y
    kernel_sobel_y = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])
                               
    return convolucion_generica(imagen, kernel_sobel_y)

def suavizado_Gaussiano(imagen):

    # Aplicar filtro Gaussiano
    kernel_gaussiano = (1/16) * np.array([[1, 2, 1],
                                          [2, 4, 2],
                                          [1, 2, 1]])
    
    return convolucion_generica(imagen, kernel_gaussiano)

def gradiente_Sobel(imagen):
    
    # Gradiente Sobel en X y Y
    conv_X = filtro_Sobel_X(imagen)
    conv_Y = filtro_Sobel_Y(imagen)
    
    # Magnitud del gradiente
    G = np.hypot(conv_X, conv_Y)        
    G = G / G.max() * 255
    
    # Dirección del gradiente
    theta = np.arctan2(conv_X, conv_Y)  
    
    return (G, theta)

def supresion_no_maxima(G, theta):

    # Establecer dirección del borde
    h, w = G.shape
    Z = np.zeros((h,w), dtype=np.float32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
    # Dejar sólo máximos locales
    for i in range(1,h-1):
        for j in range(1,w-1):
            q, r = 255, 255

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0
    
    return Z

def umbralizacion_con_histeresis(img, low, high):

    # Clasificar bordes débiles y fuertes
    strong = 255
    weak = 75
    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    # Conexión de bordes débiles a fuertes
    h, w = img.shape
    for i in range(1,h-1):
        for j in range(1,w-1):
            if res[i,j] == weak:
                if np.any(res[i-1:i+2, j-1:j+2] == strong):
                    res[i,j] = strong
                else:
                    res[i,j] = 0
    
    return res

def detector_de_Canny(imagen, low_threshold=50, high_threshold=100):

    # 1. Suavizado Gaussiano
    salida = suavizado_Gaussiano(imagen)
    imagen_suave = np.clip(np.abs(salida), 0, 255).astype(np.uint8)
    
    # 2. Gradientes con Sobel
    G, theta = gradiente_Sobel(imagen_suave)
    
    # 3. Supresión no máxima
    nms = supresion_no_maxima(G, theta)
    
    # 4. Umbralización con histéresis
    salida_final = umbralizacion_con_histeresis(nms, low_threshold, high_threshold)
    
    return salida_final

def filtro_Laplaciano(imagen):

    # Aplicar filtro Laplaciano
    kernel_laplaciano = np.array([[ 0, -1,  0],
                                  [-1,  4, -1],
                                  [ 0, -1,  0]])

    return convolucion_generica(imagen, kernel_laplaciano)
