import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

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
    
    # Normalizar valores al rango válido
    salida = np.clip(salida, 0, 255).astype(np.uint8)
    return salida

def filtro_Sobel_X(imagen_gray):

    kernel_sobel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    return convolucion_generica(imagen_gray, kernel_sobel_x)

def filtro_Sobel_Y(imagen_gray):

    kernel_sobel_y = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])
    return convolucion_generica(imagen_gray, kernel_sobel_y)

def sobel_filters(img):
    
    G = np.hypot(Gx, Gy)        
    G = G / G.max() * 255       
    theta = np.arctan2(Gy, Gx)  
    
    return (G, theta)

def non_max_suppression(G, theta):

    h, w = G.shape
    Z = np.zeros((h,w), dtype=np.float32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
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

def threshold(img, low, high):

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

def canny_detector(imagen_gray, low_threshold=50, high_threshold=100, sigma=1.4):

    # 1. Suavizado
    imagen_suave = gaussian_filter(imagen_gray, sigma=sigma)
    
    # 2. Gradientes con Sobel
    G, theta = sobel_filters(imagen_suave)
    
    # 3. Supresión no máxima
    nms = non_max_suppression(G, theta)
    
    # 4. Umbral doble + histéresis
    salida = threshold(nms, low_threshold, high_threshold)
    
    return salida
