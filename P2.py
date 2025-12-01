
### TRABAJO PRACTICO 2 - PROBLEMA 2 ###
### DETECCIÓN DE PATENTES ###


import cv2
import numpy as np
import matplotlib.pyplot as plt

# A) DETECTAR PATENTE EN LA IMAGEN

def detectar_patente(ruta):
    img = cv2.imread(ruta)
    if img is None:
        print(f"No se pudo leer la imagen: {ruta}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, kernel_rect)

    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_small = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    height_img, width_img = img.shape[:2]
    area_img = height_img * width_img 

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(max(h,1))

        if area < 800:
            continue
        if not (1.8 < ar < 9.0):
            continue
        if y + h < int(height_img * 0.30):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) < 3:
            continue

        candidatos.append((x,y,w,h,area,ar))

    if not candidatos:
        print("No se detectaron candidatos con los filtros base.")
        return None

    def energia_vertical_norm(rec):
        gray_r = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_r, cv2.CV_64F, 1, 0, ksize=3)
        sval = np.abs(sobelx).sum()
        return sval / (rec.shape[0] * rec.shape[1] + 1e-6)

    evaluados = []
    for (x,y,w,h,area,ar) in candidatos:
        rec = img[y:y+h, x:x+w]
        ev_norm = energia_vertical_norm(rec)
        pos_score = (y + h/2) / height_img
        ar_penalty = 1 - min(0.9, abs(ar - 4) / 4)
        area_score = np.log(area + 1) / np.log(width_img*height_img + 1)
        score = ev_norm * 0.6 + pos_score * 0.2 + ar_penalty * 0.15 + area_score * 0.05

        evaluados.append({
            "rect": (x,y,w,h),
            "ev": float(ev_norm),
            "pos": float(pos_score),
            "ar": float(ar),
            "area": int(area),
            "score": float(score)
        })

    evaluados = sorted(evaluados, key=lambda x: x["score"], reverse=True)

    print("=== Diagnóstico de candidatos (top 10) ===")
    for i, e in enumerate(evaluados[:10]):
        x,y,w,h = e["rect"]
        print(f"{i+1}: score={e['score']:.4f}, ev={e['ev']:.4f}, pos={e['pos']:.3f}, "
              f"ar={e['ar']:.2f}, area={e['area']}, rect={(x,y,w,h)}")

    img_annot = img.copy()
    for i, e in enumerate(evaluados):
        x,y,w,h = e["rect"]
        s = e["score"]
        ev = e["ev"]
        color = (0, int(min(255, 255 * s)), int(min(255, 255 * (1 - s))))
        cv2.rectangle(img_annot, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img_annot, f"{i+1}:{s:.2f}", (x, max(y-6,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img_annot, f"ev:{ev:.2f}", (x, y+h+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


    best = evaluados[0] # el mejor candidato
    x,y,w,h = best["rect"]
    patente = img[y:y+h, x:x+w]

    # recorte del borde interno 
    ph, pw = patente.shape[:2]
    margin = max(2, int(min(ph, pw) * 0.02))   # margen adaptativo
    patente = patente[margin:ph-margin, margin:pw-margin]


    return patente, img   




# B) DETECTAR CARACTERES EN LA PATENTE 




def detectar_caracteres(patente):
    h, w = patente.shape[:2]
    scale = 3
    img = cv2.resize(patente, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Binarización
    blur = cv2.GaussianBlur(gray, (1,1), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Separación de letras pegadas 
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    separated = cv2.erode(thresh, kernel_sep, iterations=1)

    # Pequeñas aperturas 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    clean = separated


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)

    # Calcular medidas globales para filtros adaptativos
    alturas = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
    anchos  = [stats[i, cv2.CC_STAT_WIDTH]  for i in range(1, num_labels)]
    areas   = [stats[i, cv2.CC_STAT_AREA]   for i in range(1, num_labels)]

    altura_med = np.median(alturas)
    ancho_med  = np.median(anchos)

    caracteres = []
    vis = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 50: 
            continue

        aspect = h / float(w + 1e-6)

        # Filtros optimizados para patentes de formato viejo (AAA 123)
        if not (0.9 <= aspect <= 6.5):
            continue

        if not (altura_med*0.4 <= h <= altura_med*2.5):
            continue

        if not (ancho_med*0.4 <= w <= ancho_med*3):
            continue

        # Guardar caracter
        crop = img[y:y+h, x:x+w]
        caracteres.append({"x": x, "crop": crop})

        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)

    # Ordenar
    caracteres.sort(key=lambda c: c["x"])

    return caracteres, thresh, vis


def plot_pipeline(original, patente, thresh, vis, caracteres):
    pasos_fijos = 4   # original, patente, thresh, vis
    cant_chars = len(caracteres)
    total = pasos_fijos + cant_chars

    cols = min(total, 5)
    rows = int(np.ceil(total / cols))

    fig = plt.figure(figsize=(4*cols, 4*rows))

    idx = 1

    # Imagen original
    ax = fig.add_subplot(rows, cols, idx)
    ax.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax.set_title("Original")
    ax.axis("off")
    idx += 1

    # Patente recortada
    ax = fig.add_subplot(rows, cols, idx)
    ax.imshow(cv2.cvtColor(patente, cv2.COLOR_BGR2RGB))
    ax.set_title("Patente")
    ax.axis("off")
    idx += 1

    # Binarización 
    ax = fig.add_subplot(rows, cols, idx)
    ax.imshow(thresh, cmap="gray")
    ax.set_title("Binarizada")
    ax.axis("off")
    idx += 1

    # Componentes detectados 
    ax = fig.add_subplot(rows, cols, idx)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title("Componentes")
    ax.axis("off")
    idx += 1

    # Caracteres detectados 
    for c in caracteres:
        ax = fig.add_subplot(rows, cols, idx)
        ax.imshow(c["crop"], cmap="gray")
        ax.set_title("Char")
        ax.axis("off")
        idx += 1

    plt.tight_layout()
    plt.show()


rutas = [
    f"D:/FABRO/TUIA/PROCESAMIENTO DE IMAGENES/TP 2/img{i:02}.png"
    for i in range(1, 13)
]

for ruta in rutas:
    original_patente, original_img = detectar_patente(ruta)
    caracteres, thresh, vis = detectar_caracteres(original_patente)
    plot_pipeline(original_img, original_patente, thresh, vis, caracteres)
