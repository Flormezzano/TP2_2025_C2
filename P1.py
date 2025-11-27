
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
img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)
# mostrar(img_blur, "Gris + Blur Gaussiano")


# ====================================================
#   CANNY + MORFOLOGÍA (BORDES)
# ====================================================
# Detección de bordes con Canny
# Busca cambios bruscos de intensidad. Se usan umbrales bajos (20,70)
# para capturar bordes débiles y lograr contornos más continuos.

edges = cv2.Canny(img_blur, 20, 70)
# mostrar(edges, "Detección de bordes Canny")

# Dilatación de bordes
# Agranda los bordes detectados para que queden más gruesos,
# facilitando el cierre de figuras y evitando cortes.
kern_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edges_thick = cv2.dilate(edges, kern_small, iterations=5)
# mostrar(edges_thick, "Bordes dilatados")

# Detección de contornos externos
# Busca solo contornos principales (no huecos internos).
contours, hierarchy = cv2.findContours(
    edges_thick, 
    cv2.RETR_EXTERNAL,        # solo contornos externos
    cv2.CHAIN_APPROX_SIMPLE
)


# Filtrado de contornos por área
# Elimina objetos demasiado pequeños (ruido) o gigantes (artefactos)
area_min = 1000    # Área mínima (ajustar según tu imagen)
area_max = 5000000  # Área máxima

contours_filtered = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area_min < area < area_max:
        contours_filtered.append(cnt)

# Visualización de contornos sobre la imagen original (solo para inspección)
img_contours = img_bgr.copy()
cv2.drawContours(img_contours, contours_filtered, -1, (0,255,0), 3)
# mostrar(img_contours, "Objetos detectados con contornos", cmap=None)


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



# ====================================================
#   B) CLASIFICACIÓN Y CONTEO AUTOMÁTICO DE MONEDAS
# ====================================================

objetos = []

for cnt in contours_filtered:
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        continue

    # Centroide
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # Radio equivalente de un círculo con la misma área
    radio_eq = np.sqrt(area / np.pi)

    # Círculo mínimo que envuelve al contorno
    (x_circ, y_circ), r = cv2.minEnclosingCircle(cnt)
    area_circ = np.pi * (r ** 2)
    if area_circ == 0:
        continue

    # Roundness: qué tanto llena ese círculo
    roundness = area / area_circ

    objetos.append({
        "contour": cnt,
        "area": area,
        "perim": perim,
        "cx": cx,
        "cy": cy,
        "radio_eq": radio_eq,
        "roundness": roundness,
        "r": r
    })


UMBRAL_ROUNDNESS_MONEDA = 0.8  # ajustable


monedas = [o for o in objetos if o["roundness"] >= UMBRAL_ROUNDNESS_MONEDA]
dados   = [o for o in objetos if o["roundness"] <  UMBRAL_ROUNDNESS_MONEDA]

print(f"Monedas detectadas: {len(monedas)}")
print(f"Dados detectados:   {len(dados)}")

# ====================================================
#   K-means sobre las monedas (3 tipos)
# ====================================================

if len(monedas) > 0:
    radios = np.array([[m["radio_eq"]] for m in monedas], dtype=np.float32)

    K = 3
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    compactness, labels, centers = cv2.kmeans(
        data=radios,
        K=K,
        bestLabels=None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS
    )

    # ordenar clusters de moneda chica → grande
    centers = centers.flatten()
    orden = np.argsort(centers)

    cluster2tipo = {}
    for rank, k in enumerate(orden):
        cluster2tipo[k] = f"Moneda_{rank+1}"

    # asignar tipo a cada moneda
    for i, m in enumerate(monedas):
        cl = int(labels[i])
        m["cluster"] = cl
        m["tipo"] = cluster2tipo[cl]
else:
    print("No hay monedas para clasificar con k-means.")

# Conteo de monedas por tipo
conteo = {}
for m in monedas:
    conteo[m["tipo"]] = conteo.get(m["tipo"], 0) + 1

print("\n=== Conteo automático de monedas por tipo ===")
for tipo, n in sorted(conteo.items()):
    print(f"{tipo}: {n}")

# ====================================================
#   Visualización final: monedas por tipo + dados aparte
# ====================================================

out = segm_contours.copy()   # o img_bgr.copy()

# colores por tipo de moneda (BGR)
color_tipo = {
    "Moneda_1": (0, 255,   0),   # verde
    "Moneda_2": (0, 255, 255),   # amarillo
    "Moneda_3": (255, 0,   0),   # azul
}

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.0      # texto grande
thickness = 3

# Monedas: círculo y texto con color según tipo
for m in monedas:
    cx, cy = m["cx"], m["cy"]
    r_draw = int(m["r"])
    etiqueta = m.get("tipo", "Moneda")

    c = color_tipo.get(etiqueta, (0, 255, 0))  # default verde

    # círculo
    cv2.circle(out, (cx, cy), r_draw, c, 3)

    # texto (un poco abajo del centro para que se vea)
    cv2.putText(out, etiqueta,
                (cx - 45, cy + 10),
                font, font_scale, c, thickness, cv2.LINE_AA)

# Dados: círculo rojo + texto grande "Dado"
for d in dados:
    cx, cy = d["cx"], d["cy"]
    r_draw = int(d["r"])

    cv2.circle(out, (cx, cy), r_draw, (0, 0, 255), 3)
    cv2.putText(out, "Dado",
                (cx - 40, cy + 10),
                font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

mostrar(out, "Monedas por tipo (color) y dados (rojo)", cmap=None)


