# Marcela Melgar
# 20200487

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import cvlib
import argparse

def imgview(img,filename):

    # asegurar que el array solo contenga datos numericos
    img = img.astype(np.uint8)

    # Mostrar imagen
    visualizacion = cv.imshow('Imagen', img)

    # imprimir imagen en consola
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('IMAGE VIEW')
    plt.axis('on')  # Ocultar ejes
    plt.show()

    try:
        # Guarda la imagen con el nombre especificado
        plt.savefig(filename)

        # Cierra la figura (opcional, dependiendo de tu flujo de trabajo)
        plt.close()

        print(f"La imagen se ha guardado como '{filename}' con éxito.")
    except Exception as e:
        print(f"Error al guardar la imagen: {str(e)}")
    return visualizacion

parser = argparse.ArgumentParser(description='Ingreso de parametros desde consola')

# Agregar argumentos
parser.add_argument('arg1', type=str, help='imagen de entrada')
parser.add_argument('arg2', type=str, help='imagen de salida')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()


def imgpad(img, r):
    
    """
    Rellena el alrededor de la imagen ceros de un ancho especificado r.

    Parámetros:
    img (numpy.ndarray): La imagen de entrada que se va a rellenar.
    r (int): El ancho del borde (número de píxeles) que se agregará alrededor de la imagen.

    Devuelve:
    numpy.ndarray: la matriz de pixeles de la imagen rellenada con un borde de ancho r.
    """

    height, width = img.shape

    new_h = height + 2 * r
    new_w = width + 2 * r
    padded = np.zeros((new_h, new_w), dtype=np.uint8)

    padded[r:r+height, r:r+width] = img

    padded.tolist()
    for i in padded.tolist():
        print(i)


image_path = args.arg1
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
if img.dtype != np.uint8:
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
thresh_val = 165
thresh = cv.threshold(img, thresh_val, 255, cv.THRESH_BINARY)[1]

if img is None:
    print("Error: Failed to read the image.")
else:
    print("entrada:")
    img.tolist()
    for i in img.tolist():
        print(i)
    print("salida:")
    imgpad(img, 1)

print("\n-------------------------------------\n")

def connected_c(img):

    """
    Realiza la segmentación de componentes conectados en una imagen binaria.

    Parámetros:
    img (numpy.ndarray): Imagen binaria de entrada en formato NumPy.

    Devuelve:
    numpy.ndarray: Imagen etiquetada donde cada componente conectado
                  tiene una etiqueta distinta.

    Esta función utiliza una operación morfológica para eliminar ruido en la imagen
    binaria de entrada y luego realiza la segmentación de componentes conectados
    utilizando 8-conectividad. Las etiquetas se asignan a los componentes conectados
    y se resuelve la equivalencia entre etiquetas vecinas.

    Nota:
    - La imagen binaria de entrada debe estar en formato NumPy.
    - La imagen de salida contiene etiquetas numéricas que representan los componentes
      conectados. La etiqueta 0 corresponde al fondo.
    """

    connectivity_kernel = np.array([[255, 255, 255],
                                    [255, 255, 255],
                                    [255, 255, 255]], dtype=np.uint8)

    img = cv.morphologyEx(img, cv.MORPH_OPEN, connectivity_kernel)

    num_labels, labels = cv.connectedComponents(img, connectivity=8)

    equivalence = np.arange(num_labels)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] != 0:
                neighbors = []

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                            neighbor_label = labels[ny, nx]
                            if neighbor_label != 0:
                                neighbors.append(neighbor_label)

                if neighbors:
                    min_neighbor = min(neighbors)
                    labels[y, x] = min_neighbor

                    for neighbor in neighbors:
                        if neighbor != min_neighbor:
                            equivalence[neighbor] = min_neighbor

    for label in range(1, num_labels):
        labels[labels == label] = equivalence[label]

    return labels

print(connected_c(thresh).tolist())

print("\n-------------------------------------\n")

import random
def labelview(labels):

    """
    Visualiza las etiquetas en una imagen en color.

    Parámetros:
    labels (numpy.ndarray): Matriz de etiquetas numéricas.

    Esta función toma una matriz de etiquetas numéricas y crea una representación visual
    de las etiquetas en una imagen en color. Cada etiqueta se asigna a un color aleatorio
    para resaltar las regiones etiquetadas en la imagen.

    - Las etiquetas numéricas deben ser valores enteros no negativos, donde 0 generalmente representa el fondo.
    """

    unique_labels = np.unique(labels) 
    label_colors = {} 

    for label in unique_labels:
        if label == 0:
            label_colors[label] = [0, 0, 0]
        else:

            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            label_colors[label] = color

    colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label = labels[i, j]
            colored_labels[i, j] = label_colors[label]

    imgview(colored_labels, args.arg2)

# labels = np.array([[0, 1, 1, 0, 0],
#                    [0, 2, 2, 2, 0],
#                    [0, 0, 2, 0, 3],
#                    [0, 0, 0, 0, 3]])

labelview(connected_c(thresh))

