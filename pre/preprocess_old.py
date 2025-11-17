import numpy as np
import cv2 as cv


def rms(data, k=0.1) -> float:
    b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
    b3, b4 = int((1 - k) * data.shape[0]), int((1 - k) * data.shape[1])

    upper_left = np.mean(data[:b1, :b2].flatten() ** 2)
    upper_right = np.mean(data[b3:, :b2].flatten() ** 2)
    down_left = np.mean(data[:b1, b4:].flatten() ** 2)
    down_right = np.mean(data[b3:, b4:].flatten() ** 2)
    noise = np.mean([upper_left, upper_right, down_left, down_right])
    return np.sqrt(noise)


def preprocess(map2d: np.ndarray, lognorm: bool = False) -> np.ndarray:
    shape = (128, 128)
    map2d = cv.resize(map2d, shape)

    if lognorm:
        vmin = rms(map2d) * 3
        map2d = np.clip(map2d, vmin, np.inf)
        vmax = map2d.max()
        map2d_clipped = np.clip(map2d, a_min=vmin, a_max=vmax)
        map2d_log = np.log(map2d_clipped)
        im_norm = (map2d_log - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
        # FIXME: this does not work
        min_val = im_norm.min()
        max_val = im_norm.max()
        im_norm = (im_norm - min_val) / (max_val - min_val) * 255
        return im_norm.astype(np.uint8)
    return map2d
