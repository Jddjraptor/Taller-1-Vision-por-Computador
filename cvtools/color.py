import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def RGB_HSV(rgb):
    
    #Normalizar
    rgb = rgb.astype("float32") / 255.0
    R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
    
    #Valores Intermedios
    Cmax = np.max(rgb, axis=-1)
    Cmin = np.min(rgb, axis=-1)
    delta = Cmax - Cmin
    
    # Hue
    H = np.zeros_like(Cmax)
    mask = delta != 0
    idx = (Cmax == R) & mask
    H[idx] = (60 * ((G[idx] - B[idx]) / delta[idx]) % 360)
    idx = (Cmax == G) & mask
    H[idx] = (60 * ((B[idx] - R[idx]) / delta[idx] + 2))
    idx = (Cmax == B) & mask
    H[idx] = (60 * ((R[idx] - G[idx]) / delta[idx] + 4))
    
    # Saturation
    S = np.zeros_like(Cmax)
    S[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]
    
    # Value
    V = Cmax
    
    return np.stack([H, S, V], axis=-1)

def RGB_LAB(rgb):
    
    # Normalizar
    rgb = rgb.astype("float32") / 255.0
    
    # Linealizar
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    
    # Matriz sRGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    XYZ = np.dot(rgb, M.T)
    
    # Normalizar (D65)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = XYZ[...,0] / Xn
    Y = XYZ[...,1] / Yn
    Z = XYZ[...,2] / Zn
    
    # Función f(t)
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787*t + 16/116)
    
    # L,a,b
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    
    return np.stack([L, a, b], axis=-1)

def histograma_de_colores(imagen_rgb):

    # Separar canales
    R = imagen_rgb[:,:,0].ravel()
    G = imagen_rgb[:,:,1].ravel()
    B = imagen_rgb[:,:,2].ravel()
    
    # Histograma con 256 bins (0-255)
    hist_r, _ = np.histogram(R, bins=256, range=(0,256))
    hist_g, _ = np.histogram(G, bins=256, range=(0,256))
    hist_b, _ = np.histogram(B, bins=256, range=(0,256))
    
    # Graficar histograma
    
    plt.figure(figsize=(10,5))
    bins = np.arange(256)
    
    plt.plot(bins, hist_r, color='red', label='Rojo')
    plt.plot(bins, hist_g, color='green', label='Verde')
    plt.plot(bins, hist_b, color='blue', label='Azul')
    
    plt.title("Histograma de Colores")
    plt.xlabel("Intensidad (0-255)")
    plt.ylabel("Frecuencia de píxeles")
    plt.legend()
    plt.show()

def cuantizacion_simple(imagen_rgb, niveles_por_canal):

    # Calcular el tamaño de paso
    step = 256 // niveles_por_canal
    
    # Cuantizar cada canal
    imagen_cuantizada = (imagen_rgb // step) * step + step // 2
    imagen_cuantizada = np.clip(imagen_cuantizada, 0, 255).astype(np.uint8)
    
    return imagen_cuantizada

def reducir_peso_imagen(ruta_imagen, niveles_por_canal, ruta_salida="imagen_reducida.png"):

    # Cargar imagen
    imagen = Image.open(ruta_imagen).convert("RGB")
    imagen_rgb = np.array(imagen)
    
    # Cuantización uniforme
    imagen_cuantizada = cuantizacion_simple(imagen_rgb, niveles_por_canal)
    
    # Guardar imagen reducida
    img_out = Image.fromarray(imagen_cuantizada)
    img_out.save(ruta_salida, optimize=True)
    
    # Calcular tamaño en KB
    tamaño_kb = os.path.getsize(ruta_salida) / 1024
    
    return tamaño_kb
