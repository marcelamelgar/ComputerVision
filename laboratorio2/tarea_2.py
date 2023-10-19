# Esteban Samayoa - 20200188
# Marcela Melgar - 20200487

import sys
import cv2 as cv
import numpy as np
import time 


def sharpen(img):
    """
    Aplica el efecto de enfoque nítido a una imagen.

    Args:
        img (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen con el efecto de enfoque nítido aplicado.
    """
    # Define el kernel para el filtro de enfoque nítido
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    # Obtén las dimensiones de la imagen
    height, width, channels = img.shape

    # Inicializa una nueva imagen para el resultado
    img_sharpened = np.zeros((height, width, channels), dtype=np.uint8)

    # Aplica el filtro manualmente
    for channel in range(channels):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Calcula el nuevo valor de píxel aplicando el kernel
                pixel_value = 0
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        pixel_value += img[i + m, j + n, channel] * kernel[m + 1, n + 1]

                # Asegúrate de que el valor esté en el rango [0, 255]
                img_sharpened[i, j, channel] = np.clip(pixel_value, 0, 255)

    return img_sharpened


# Nombre del archivo de video .mp4
video_file = 'leyendo.mp4'

# Abre el archivo de video
cap = cv.VideoCapture(video_file)

# Verifica que el archivo de video esté abierto
if not cap.isOpened():
    print("Error al abrir el archivo de video")
    sys.exit()
print('si abrio el video')

start_time = time.time()  # Registra el tiempo de inicio

# Inicializa un contador de fotogramas
frame_count = 0

# Obtiene un fotograma, aplica el procesamiento y muestra los resultados
while True:
    ret, im_rgb = cap.read()

    if ret:
        # Aplica el efecto HDR
        sharpen_image = sharpen(im_rgb)
        frame_count += 1

        win0 = 'Original'
        win1 = 'Efecto Sharpen'

        r, c = im_rgb.shape[0:2]
        resize_factor = 2

        R = int(r // resize_factor)
        C = int(c // resize_factor)
        win_size = (C, R)

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)

        cv.resizeWindow(win0, win_size)
        cv.resizeWindow(win1, win_size)

        cv.imshow(win0, im_rgb)
        cv.imshow(win1, sharpen_image)

        # Alinea las ventanas
        cv.moveWindow(win0, 0, 0)
        cv.moveWindow(win1, C, 0)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

end_time = time.time()  # Registra el tiempo de finalización

# Calcula el tiempo transcurrido
elapsed_time = end_time - start_time

print(f"Tiempo total: {elapsed_time:.2f} segundos")
print(f"Número total de fotogramas procesados: {frame_count}")

# Limpieza antes de salir
cap.release()
cv.destroyAllWindows()