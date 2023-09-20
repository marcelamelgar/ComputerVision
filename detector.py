# Marcela Melgar
# 20200487

import argparse
import cv2 as cv
import numpy as np

# Inicializar una lista para almacenar áreas de contornos
bords = []

def detect_license_plate(image_path):
    """
    Cargar una imagen desde la ubicación especificada.

    Args:
        image_path (str): La ubicación de la imagen.

    Returns:
        image: La imagen cargada.
    """
    image = cv.imread(image_path)

    if image is None:
        print("No se pudo cargar la imagen.")
    else:
        print("Imagen cargada correctamente.")

    return image

def procesamiento(img):
    """
    Procesar la imagen para detectar contornos y áreas de interés.

    Args:
        img: La imagen de entrada.

    Returns:
        cropped_image: Imagen procesada con áreas de interés resaltadas.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convertir a escala de grises
    gray = cv.blur(gray, (3, 3))
    bordes = cv.Canny(gray, 50, 100)
    bordes = cv.dilate(bordes, None, iterations=1)

    contornos, _ = cv.findContours(bordes, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv.contourArea(c)

        if 9000 < area:  # Ajustar el umbral del área según sea necesario
            bords.append(area)
            print(area)
            cts = cv.drawContours(img.copy(), [c], -1, (0, 255, 0), 2)  # Dibujar en una copia de la imagen original

            inner_contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=c[0][0])

            for inner_contour in inner_contours:
                inner_area = cv.contourArea(inner_contour)

                if inner_area > 1000:  # Ajustar el umbral del área interna según sea necesario
                    cts = cv.drawContours(cts, [inner_contour], -1, (0, 0, 255), 2)

    maxi = min(bords)

    if maxi:
        mask = np.zeros_like(gray)
        for c in contornos:
            area = cv.contourArea(c)
            if area == maxi:
                mask = cv.drawContours(mask, [c], -1, 255, thickness=cv.FILLED)

        cropped_image = cv.bitwise_and(img, img, mask=mask)

    else:
        print("No se encontraron contornos.")

    return cropped_image

import cv2 as cv
import numpy as np

def letter_detector(img):
    """
    Detectar y agrupar contornos en la imagen por altura de sus rectángulos delimitadores.

    Args:
        img: La imagen de entrada.

    Returns:
        imgcont: Imagen con contornos agrupados y resaltados.
    """
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    filtered_image = cv.medianBlur(imgray, 5)

    ret, imgbin = cv.threshold(filtered_image, 120, 255, cv.THRESH_BINARY)

    mode = cv.RETR_TREE  # Modo de recuperación de contornos
    method = [cv.CHAIN_APPROX_NONE, cv.CHAIN_APPROX_SIMPLE]  # Método de aproximación de contornos
    contours, hierarchy = cv.findContours(imgbin, mode, method[1])

    color = (0, 255, 0)  # Color para dibujar contornos (r, g, b)
    thickness = 3
    imgcont = img.copy()

    # Ordenar los contornos por la altura de sus rectángulos delimitadores en orden ascendente
    contours = sorted(contours, key=lambda x: cv.boundingRect(x)[3])

    # Inicializar una lista para almacenar contornos agrupados
    grouped_contours = []

    # Agrupar contornos con la misma o similar altura (dentro de una tolerancia)
    current_height = None
    current_group = []

    height_tolerance = 5  # Tolerancia de altura para agrupar

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        if current_height is None:
            current_height = h

        if abs(h - current_height) <= height_tolerance:
            current_group.append(contour)
        else:
            if len(current_group) > 1:
                grouped_contours.append(current_group)
            current_group = [contour]
            current_height = h

    # Verificar si hay contornos restantes
    if len(current_group) > 1:
        grouped_contours.append(current_group)

    # Dibujar rectángulos alrededor de los contornos agrupados
    for group in grouped_contours:
        for contour in group:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(imgcont, (x, y), (x + w, y + h), color, thickness)

    cv.imshow('Contornos Agrupados', imgcont)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return imgcont

def main():
    parser = argparse.ArgumentParser(description="Detecta matrículas en una imagen.")
    parser.add_argument("--p", dest="image_path", help="Ubicación de la imagen a analizar")

    args = parser.parse_args()

    if args.image_path:
        img = detect_license_plate(args.image_path)
        img_processed = procesamiento(img)
        letter_detector(img_processed)
    else:
        print("Debes especificar la ubicación de la imagen con --p.")

if __name__ == "__main__":
    main()
