import numpy as np
import ctypes
import os
import cv2

# so_path = os.path.join(os.path.dirname(__file__), 'libtest.so')

# lib = ctypes.cdll.LoadLibrary(so_path)
# c_float_p = ctypes.POINTER(ctypes.c_float)


def extract_points(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    data_p = image.astype(np.float32).ctypes.data_as(c_float_p)

    result = np.zeros((2500, 2)).astype(np.float32)
    result_p = result.ctypes.data_as(c_float_p)

    point_num = lib.main(640, 480, data_p, 2000, 2500, result_p)
    points = result[:int(point_num), :]

    # for (y, x) in points:
    #     image = cv2.circle(image, (x, y), 2, (255, 0, 255), -1)
    #
    # cv2.imshow("ii", image)
    # cv2.waitKey()
    return points


class PixelSelector:
    def __init__(self):
        # self.so_path = os.path.join(os.path.dirname(__file__), 'libtest.so')

        self.so_path = './datasets/libtest.so'
        self.lib = ctypes.cdll.LoadLibrary(self.so_path)
        self.c_float_p = ctypes.POINTER(ctypes.c_float)

    def extract_points(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (640, 480))
        data_p = image.astype(np.float32).ctypes.data_as(self.c_float_p)

        result = np.zeros((2500, 2)).astype(np.float32)
        result_p = result.ctypes.data_as(self.c_float_p)

        point_num = self.lib.main(640, 480, data_p, 2000, 2500, result_p)
        points = result[:int(point_num), :]

        return points



