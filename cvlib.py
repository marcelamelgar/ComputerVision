# cvlib.py

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def imgreed(filename):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    return img

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

def hist(img):

    # asegurar que el array solo contenga datos numericos
    img = img.astype(np.uint8)

    # Crear una figura con dos subplots: uno para la imagen y otro para el histograma
    fig, (ax_imagen, ax_histograma) = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen en el primer subplot usando cv2.imshow()
    ax_imagen.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax_imagen.set_title('Imagen')
    ax_imagen.axis('off')  # Ocultar ejes para la imagen

    # Calcular el histograma de la imagen
    histograma = cv.calcHist([img], [0], None, [256], [0, 256])

    # Mostrar el histograma en el segundo subplot
    valores_pixeles = np.arange(256)
    ax_histograma.plot(valores_pixeles, histograma, color='black')
    ax_histograma.set_title('Histograma')
    ax_histograma.set_xlim([0, 256])
    ax_histograma.set_xlabel('Pixel Value')
    ax_histograma.set_ylabel('Pixel Count')
    ax_histograma.axis('on')

    # Mostrar la figura
    plt.tight_layout()
    plt.show()

img = cv.imread('subimage.pgm', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('baboon.tiff', cv.IMREAD_GRAYSCALE)
# hist(img)

def imgcmp(img1, img2):

    # Asegurarnos de que las imágenes sean del tipo uint8 y estén en el rango [0, 255]
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    # Crear una figura con dos subplots: uno para la imagen1 y otro para la imagen2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar imagen1 en el primer subplot
    ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    ax1.set_title('Imagen 1')
    ax1.axis('on')  # Ocultar ejes para la imagen1

    # Mostrar imagen2 en el segundo subplot
    ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    ax2.set_title('Imagen 2')
    ax2.axis('on')  # Ocultar ejes para la imagen2

    # Mostrar la figura con ambas imágenes lado a lado
    plt.tight_layout()
    plt.show()

#imgcmp(img, img2)