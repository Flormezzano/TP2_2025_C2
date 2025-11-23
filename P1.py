
### TRABAJO PRACTICO 2 - PROBLEMA 1 ###
### DETECCIÓN Y CLASIFICACIÓN ###

import cv2
import numpy as np
import matplotlib.pyplot as plt 

def mostrar(img, titulo="", cmap='gray'):
    plt.figure(figsize=(5,5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    plt.title(titulo)
    plt.axis('off')
    plt.show()

ruta = "C:\\Users\\Usuario\\OneDrive\\Documentos\\TUIA\\4to Cuatri\\PDI\\TP2\\monedas.jpg"
img_bgr = cv2.imread(ruta)

if img_bgr is None:
    raise FileNotFoundError("No se pudo leer la imagen. Revisá la ruta a monedas.jpg")

# mostrar(img_bgr, "Imagen original", cmap=None)

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# mostrar(img_gray, "Imagen en escala de grises")

# Filtro Gaussiano para reducir ruido
img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
# mostrar(img_blur, "Gris + Blur Gaussiano")


# ====================================================
#   CANNY + MORFOLOGÍA (BORDES)
# ====================================================
# Detección de bordes con Canny
# Busca cambios bruscos de intensidad. Se usan umbrales bajos (20,70)
# para capturar bordes débiles y lograr contornos más continuos.

edges = cv2.Canny(img_blur, 20, 70)
mostrar(edges, "Detección de bordes Canny")

# Dilatación de bordes
# Agranda los bordes detectados para que queden más gruesos,
# facilitando el cierre de figuras y evitando cortes.
kern_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edges_thick = cv2.dilate(edges, kern_small, iterations=5)
mostrar(edges_thick, "Bordes dilatados")

# Detección de contornos externos
# Busca solo contornos principales (no huecos internos).
contours, hierarchy = cv2.findContours(
    edges_thick, 
    cv2.RETR_EXTERNAL,        # solo contornos externos
    cv2.CHAIN_APPROX_SIMPLE
)


# Filtrado de contornos por área
# Elimina objetos demasiado pequeños (ruido) o gigantes (artefactos)
area_min = 50    # Área mínima (ajustar según tu imagen)
area_max = 5000000  # Área máxima

contours_filtered = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area_min < area < area_max:
        contours_filtered.append(cnt)

# Visualización de contornos sobre la imagen original (solo para inspección)
img_contours = img_bgr.copy()
cv2.drawContours(img_contours, contours_filtered, -1, (0,255,0), 3)
mostrar(img_contours, "Objetos detectados con contornos", cmap=None)


# Creación de una máscara binaria vacía
# donde se dibujarán las regiones correspondientes a los objetos.
mask_contours = np.zeros_like(img_gray)


# Rellenar contornos detectados
# thickness = -1 rellena toda la forma con blanco.
cv2.drawContours(mask_contours, contours_filtered, contourIdx=-1, color=255, thickness=-1)
mostrar(mask_contours, "Máscara con contornos filtrados")

# Cierre morfológico
# (dilatación seguida de erosión) para suavizar bordes y tapar pequeños huecos.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_closed = cv2.morphologyEx(mask_contours, cv2.MORPH_CLOSE, kernel, iterations=2)
# mostrar(mask_closed, "Máscara tras cierre morfológico")

# Aplicar la máscara cerrada sobre la imagen original
segm_contours = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_closed)
mostrar(segm_contours, "Objetos segmentados", cmap=None)

