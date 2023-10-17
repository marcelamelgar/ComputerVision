import sys
import cv2 as cv
import numpy as np
import sharpen_cython  # Importa el módulo Cython

#Obtiene el controlador de la cámara
device_id = 0
cap = cv.VideoCapture(device_id)


if not cap.isOpened():
    print("Error al abrir la captura de video")
    sys.exit()


while True:
    ret, im_rgb = cap.read()

    if ret:
        # Aplica el efecto HDR utilizando la función Cython
        sharpen_image = sharpen_cython.sharpen_cython(im_rgb)

        # Crea ventanas
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

        # Salir con 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()