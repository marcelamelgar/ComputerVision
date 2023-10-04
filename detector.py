# Marcela Melgar
# 20200487

import argparse
import cv2 as cv
import numpy as np
import joblib

# Inicializar una lista para almacenar áreas de contornos
bords = []

def detect_license_plate(image_path):
    """
    Carga una imagen desde la ubicación especificada.

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
    Procesa la imagen para detectar contornos y áreas de interés.

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

    # en base al area de la placa, encuentra los bordes
    for c in contornos:
        area = cv.contourArea(c)

        if 9000 < area:  
            bords.append(area)
            cts = cv.drawContours(img.copy(), [c], -1, (0, 255, 0), 2)  # Dibujar en una copia de la imagen original

            inner_contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=c[0][0])

            for inner_contour in inner_contours:
                inner_area = cv.contourArea(inner_contour)

                if inner_area > 1000: 
                    cts = cv.drawContours(cts, [inner_contour], -1, (0, 0, 255), 2)

    maxi = min(bords)
    # busca el borden mas grande para que sea el borde de la placa
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

def imagen_contorno(img):
    """
    Procesa la imagen para detectar contornos y áreas de interés.

    Args:
        img: La imagen de entrada.

    Returns:
        img_with_contour: Imagen con el último contorno dibujado.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convertir a escala de grises
    gray = cv.blur(gray, (3, 3))
    bordes = cv.Canny(gray, 50, 100)
    bordes = cv.dilate(bordes, None, iterations=1)

    contornos, _ = cv.findContours(bordes, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # en base al área de la placa, encuentra los bordes
    bords = []
    for c in contornos:
        area = cv.contourArea(c)

        if 9000 < area:  
            bords.append(area)
            cts = cv.drawContours(img.copy(), [c], -1, (0, 255, 0), 2)  # Dibujar en una copia de la imagen original

            inner_contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=c[0][0])

            for inner_contour in inner_contours:
                inner_area = cv.contourArea(inner_contour)

                if inner_area > 1000: 
                    cts = cv.drawContours(cts, [inner_contour], -1, (0, 0, 255), 2)

    maxi = max(bords, default=None)
    # busca el borde más grande para que sea el borde de la placa
    if maxi is not None:
        for c in contornos:
            area = cv.contourArea(c)
            if area == maxi:
                img_with_contour = cv.drawContours(img.copy(), [c], -1, (0, 255, 0), 2)
                break
    else:
        print("No se encontraron contornos.")
        img_with_contour = img.copy()

    return img_with_contour

def letter_detector(img):
    """
    Detecta y agrupa contornos en la imagen por altura de sus rectángulos delimitadores.

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
        
    lista_mas_larga = max(grouped_contours, key=lambda lista: len(lista))
    grouped_contours = sorted(lista_mas_larga, key=lambda contour: (cv.boundingRect(contour)[0], cv.boundingRect(contour)[1]))

    # Dibujar rectángulos alrededor de los contornos agrupados
    for group in grouped_contours:
        x, y, w, h = cv.boundingRect(group)
        cv.rectangle(imgcont, (x, y), (x + w, y + h), color, thickness)

    subimages = []

    # Iterar a través de los contornos
    for contour in grouped_contours:
        # Obtener el cuadro delimitador del contorno
        x, y, w, h = cv.boundingRect(contour)
        
        # Crear una subimagen (ROI) recortando la región de la imagen original
        subimage = imgcont[y:y+h, x:x+w]

        # Agregar la subimagen a la lista
        subimages.append(subimage)


    imagenes_binarizadas_con_borde = []

    for imagen_lista in subimages:
        # Convertir la imagen a escala de grises
        gray = cv.cvtColor(imagen_lista, cv.COLOR_BGR2GRAY)

        # Aplicar un umbral binario para hacer que el fondo sea negro y las letras sean blancas
        _, binary_image = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

        # Agregar un borde negro a la imagen binarizada
        bordered_image = cv.copyMakeBorder(binary_image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=255)
        new_size = (75, 100)
        bordered_image = cv.resize(bordered_image, new_size)
        # Agregar la imagen binarizada con el borde a la lista
        imagenes_binarizadas_con_borde.append(bordered_image)
    
    # for i, image in enumerate(imagenes_binarizadas_con_borde):
    #     cv.imshow(f'Image {i}', image)
    #     cv.waitKey(0)

    model = joblib.load("model.sav")

    if len(imagenes_binarizadas_con_borde) == 0:
        print("La lista de imágenes está vacía.")
    else:
        # Inicializar una lista para almacenar las imágenes aplanadas
        imagenes_aplanadas = []

        # Aplanar cada imagen y asegurarse de que tenga 7500 características
        for imagen in imagenes_binarizadas_con_borde:
            # Aplanar la imagen
            imagen_aplanada = imagen.flatten()

            # Asegurarse de que tenga exactamente 7500 características
            if len(imagen_aplanada) != 7500:
                print("La imagen no tiene la longitud adecuada.")
            else:
                imagenes_aplanadas.append(imagen_aplanada)

        if len(imagenes_aplanadas) == 0:
            print("No se encontraron imágenes aplanadas correctamente.")
        else:
            # Convertir la lista de imágenes aplanadas en una matriz numpy 2D
            datos_2d = np.array(imagenes_aplanadas)

            # Asegurarse de que la forma del array sea compatible con el modelo
            if datos_2d.shape[1] != 7500:
                print("La forma de los datos no es compatible con el modelo.")
            else:
                # Realizar la predicción con el modelo
                predicciones = model.predict(datos_2d)
                print(predicciones)

    resultado = ' '.join(predicciones)

    return resultado

def vista_final(original,texto):
    posicion = (60, 60)  # Las coordenadas (x, y) donde se imprimirá el texto

    # Definir el tipo de fuente, el tamaño y el color del texto
    fuente = cv.FONT_HERSHEY_SIMPLEX
    tamanio_fuente = 1
    color = (0, 255, 0)  # Color en formato BGR (verde en este ejemplo)

    # Utilizar cv2.putText para imprimir el texto en la imagen
    cv.putText(original, texto, posicion, fuente, tamanio_fuente, color, thickness=2)

    cv.imshow('Contornos Agrupados', original)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Detecta matrículas en una imagen.")
    parser.add_argument("--p", dest="image_path", help="Ubicación de la imagen a analizar")

    args = parser.parse_args()

    if args.image_path:
        img = detect_license_plate(args.image_path)
        img_processed = procesamiento(img)
        detected = letter_detector(img_processed)
        vista_final(imagen_contorno(img),detected)
    else:
        print("Debes especificar la ubicación de la imagen con --p.")

if __name__ == "__main__":
    main()
