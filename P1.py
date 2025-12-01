
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

mostrar(img_bgr, "Imagen original", cmap=None)

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
mostrar(img_gray, "Imagen en escala de grises")

# Filtro Gaussiano para reducir ruido
img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)
mostrar(img_blur, "Gris + Blur Gaussiano")


# A) CANNY + MORFOLOGÍA (BORDES)

# Detección de bordes con Canny
# Busca cambios bruscos de intensidad. 
# Se usan umbrales bajos (20,70) para capturar bordes débiles y lograr contornos más continuos.

edges = cv2.Canny(img_blur, 20, 70)
mostrar(edges, "Detección de bordes Canny")

# Dilatación de bordes
# Agranda los bordes detectados para que queden más gruesos, Facilitando el cierre de figuras y evitando cortes.
kern_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edges_thick = cv2.dilate(edges, kern_small, iterations=5)
# mostrar(edges_thick, "Bordes dilatados")

# Detección de contornos externos
# Busca solo contornos principales (no huecos internos).
contours, hierarchy = cv2.findContours(
    edges_thick, 
    cv2.RETR_EXTERNAL,       
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

# Visualización de contornos sobre la imagen original 
img_contours = img_bgr.copy()
cv2.drawContours(img_contours, contours_filtered, -1, (0,255,0), 3)
mostrar(img_contours, "Objetos detectados con contornos", cmap=None)


# Creación de una máscara binaria vacía donde se dibujarán las regiones correspondientes a los objetos.
mask_contours = np.zeros_like(img_gray)


# Rellenar contornos detectados
# thickness = -1 rellena toda la forma con blanco.
cv2.drawContours(mask_contours, contours_filtered, contourIdx=-1, color=255, thickness=-1)
mostrar(mask_contours, "Máscara con contornos filtrados")

# Cierre morfológico (dilatación seguida de erosión) para suavizar bordes y tapar pequeños huecos.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_closed = cv2.morphologyEx(mask_contours, cv2.MORPH_CLOSE, kernel, iterations=2)
mostrar(mask_closed, "Máscara tras cierre morfológico")

# Aplicar la máscara cerrada sobre la imagen original
segm_contours = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_closed)
mostrar(segm_contours, "Objetos segmentados", cmap=None)




#   B) CLASIFICACIÓN Y CONTEO AUTOMÁTICO DE MONEDAS


UMBRAL_ROUNDNESS_MONEDA = 0.8  
K = 3

# Lista donde guardamos la información de cada objeto (moneda o dado)
objetos = []

for cnt in contours_filtered:
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        continue

    # Centroide del contorno
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




# Separación monedas / dados según roundness
monedas = [o for o in objetos if o["roundness"] >= UMBRAL_ROUNDNESS_MONEDA]
dados   = [o for o in objetos if o["roundness"] <  UMBRAL_ROUNDNESS_MONEDA]

print(f"Monedas detectadas: {len(monedas)}")
print(f"Dados detectados:   {len(dados)}")


#   K-means sobre las monedas (3 tipos)

if len(monedas) > 0:
    radios = np.array([[m["radio_eq"]] for m in monedas], dtype=np.float32)

   
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

    # Ordenar clusters de moneda chica → grande
    centers = centers.flatten()
    orden = np.argsort(centers)
    
    # Mapear índice de cluster a etiqueta de tipo
    cluster2tipo = {}
    for rank, k in enumerate(orden):
        cluster2tipo[k] = f"Moneda_{rank+1}"

    # Asignar tipo a cada moneda
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


#   Visualización final: monedas por tipo + dados aparte

out = segm_contours.copy()  

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

    c = color_tipo.get(etiqueta, (0, 255, 0))  

    # círculo
    cv2.circle(out, (cx, cy), r_draw, c, 3)

    # texto 
    cv2.putText(out, etiqueta,
                (cx - 45, cy + 10),
                font, font_scale, c, thickness, cv2.LINE_AA)

for d in dados:
    cx, cy = d["cx"], d["cy"]
    r_draw = int(d["r"])

    cv2.circle(out, (cx, cy), r_draw, (0, 0, 255), 3)
    cv2.putText(out, "Dado",
                (cx - 40, cy + 10),
                font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

mostrar(out, "Monedas por tipo y dados", cmap=None)


#   C) DETECCIÓN AUTOMÁTICA DEL VALOR DE CADA DADO


# lista para guardar el valor (cantidad de puntos) de cada dado
valores_dados = []  

img_C = out.copy()

# Parámetros globales para los pips
PIP_AREA_MIN = 500   
PIP_AREA_MAX = 8000    
UMBRAL_ROUNDNESS_PIP = 0.7
UMBRAL_ASPECT_RATIO = 1.35 
    
for idx, d in enumerate(dados, start=1):
    cnt = d["contour"]

    # Bounding box del dado (rectángulo que lo encierra)
    x, y, w, h = cv2.boundingRect(cnt)

    # Recortamos ROI en gris y la máscara en esa misma región
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_mask = mask_closed[y:y+h, x:x+w]

    # Dejamos solo la zona del dado (fondo = negro)
    roi_gray_masked = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)

    # Blur + Otsu invertido para resaltar pips
    roi_blur = cv2.GaussianBlur(roi_gray_masked, (5, 5), 0)
    _, roi_bin = cv2.threshold(
        roi_blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    ## Cierre para rellenar agujeritos y apertura para limpiar ruido
    kernel_pip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi_bin_clean = cv2.morphologyEx(roi_bin, cv2.MORPH_CLOSE, kernel_pip, iterations=2)
   
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_bin_clean = cv2.morphologyEx(roi_bin_clean, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Contornos potenciales de pips
    cont_pips, _ = cv2.findContours(
        roi_bin_clean,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
          
    contornos_validos = []

    for c in cont_pips:
        area = cv2.contourArea(c)
        if not (PIP_AREA_MIN < area < PIP_AREA_MAX):
            continue

        # roundness 
        (xc, yc), r = cv2.minEnclosingCircle(c)
        if r == 0:
            continue
        area_circ = np.pi * (r ** 2)
        roundness = area / area_circ
    

        # relación de aspecto (para descartar formas muy alargadas)
        x_b, y_b, w_b, h_b = cv2.boundingRect(c)
        if w_b == 0 or h_b == 0:
            continue
        aspect_ratio = max(w_b, h_b) / min(w_b, h_b)
        

        # condición combinada: tamaño razonable + bastante circular + no muy alargado
        if (roundness >= UMBRAL_ROUNDNESS_PIP) and (aspect_ratio <= UMBRAL_ASPECT_RATIO):
            contornos_validos.append(c)
            
    # Valor del dado = cantidad de pips válidos
    num_pips = len(contornos_validos)
    # print(f"Pips detectados en el dado {idx}: {num_pips}")

    valores_dados.append(num_pips)
    d["valor"] = num_pips  

    # Visualización de los pips válidos en el ROI 
    img_cont = cv2.cvtColor(roi_bin_clean, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_cont, contornos_validos, -1, (0, 255, 0), 2)
    # mostrar(img_cont, f"Dado {idx} - Contornos de pips válidos", cmap=None)

    
# Mostrar valores por consola
print("\n=== Valores detectados en los dados ===")
for i, v in enumerate(valores_dados, start=1):
    print(f"Dado {i}: {v}")



