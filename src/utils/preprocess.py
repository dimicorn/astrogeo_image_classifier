import numpy as np
import cv2 as cv

def preprocess(map2d: np.ndarray) -> np.ndarray:
    print(map2d.dtype)
    shape = (128, 128)
    map2d = cv.resize(map2d, shape)
    return map2d