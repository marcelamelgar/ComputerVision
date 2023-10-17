import numpy as np
cimport numpy as np

def sharpen_cython(np.ndarray[np.uint8_t, ndim=3] img):
    cdef int height, width, channels
    cdef np.ndarray[np.uint8_t, ndim=3] img_sharpened
    cdef np.ndarray[np.float64_t, ndim=2] kernel

    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    img_sharpened = np.zeros((height, width, channels), dtype=np.uint8)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float64)

    for channel in range(channels):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                pixel_value = 0.0
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        pixel_value += img[i + m, j + n, channel] * kernel[m + 1, n + 1]
                img_sharpened[i, j, channel] = np.uint8(np.clip(pixel_value, 0, 255))

    return img_sharpened
