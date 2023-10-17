# Esteban Samayoa - 20200188
# Marcela Melgar - 20200487

import sys
import cv2 as cv

# Define la función de efecto HDR
def HDR(img):
    """
    Aplica el efecto HDR a una imagen.

    Args:
        img (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen con el efecto HDR aplicado.
    """
    hdr = cv.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

# Define la función de efecto de dibujo a lápiz en color
def pencil_sketch_col(img):
    """
    Aplica un efecto de dibujo a lápiz en color a una imagen.

    Args:
        img (numpy.ndarray): La imagen de entrada.

    Returns:
        numpy.ndarray: La imagen con el efecto de dibujo a lápiz en color aplicado.
    """
    sk_gray, sk_color = cv.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_color

# Obtiene el controlador de la cámara
device_id = 0
cap = cv.VideoCapture(device_id)

# Verifica que el controlador de video esté abierto
if not cap.isOpened():
    print("Error al abrir la captura de video")
    sys.exit()

# Obtiene un fotograma, aplica el procesamiento y muestra los resultados
while True:
    ret, im_rgb = cap.read()

    if ret:
        # Aplica el efecto HDR
        hdr_image = HDR(im_rgb)

        # Aplica el efecto de dibujo a lápiz en color
        color_sketch = pencil_sketch_col(im_rgb)

        # Crea ventanas
        win0 = 'Original'
        win1 = 'Efecto HDR'
        win2 = 'Dibujo a Lápiz en Color'

        r, c = im_rgb.shape[0:2]
        resize_factor = 2

        R = int(r // resize_factor)
        C = int(c // resize_factor)
        win_size = (C, R)

        cv.namedWindow(win0, cv.WINDOW_NORMAL)
        cv.namedWindow(win1, cv.WINDOW_NORMAL)
        cv.namedWindow(win2, cv.WINDOW_NORMAL)

        cv.resizeWindow(win0, (win_size[0] // 2, win_size[1] // 2))
        cv.resizeWindow(win1, win_size)
        cv.resizeWindow(win2, win_size)

        cv.imshow(win0, im_rgb)
        cv.imshow(win1, hdr_image)
        cv.imshow(win2, color_sketch)

        # Alinea las ventanas
        cv.moveWindow(win0, 0, 0)
        cv.moveWindow(win1, C, 0)
        cv.moveWindow(win2, 2 * C, 0)

        # Salir con 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Limpieza antes de salir
cap.release()
cv.destroyAllWindows()
