import numpy as np
from matplotlib.colors import LogNorm


def rms(data: np.ndarray, k: float = 0.1) -> float:
    b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
    b3, b4 = int((1-k) * data.shape[0]), int((1-k) * data.shape[1])
        
    upper_left = np.mean(data[:b1, :b2].flatten() ** 2)
    upper_right = np.mean(data[b3:, :b2].flatten() ** 2)
    down_left = np.mean(data[:b1, b4:].flatten() ** 2)
    down_right = np.mean(data[b3:, b4:].flatten() ** 2)
    noise = np.mean([upper_left, upper_right, down_left, down_right])
    return np.sqrt(noise)

im = np.random.rand(128, 128)
vmin = rms(im)
vmax = im.max()
n = LogNorm(vmin=vmin, vmax=vmax, clip=True)
im_norm1 = n(im)

im_clipped = np.clip(im, a_min=vmin, a_max=vmax)
im_log = np.log(im_clipped)
im_norm2 = (im_log - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
print(np.allclose(im_norm1, im_norm2))
print(im_norm1[:2, :2])
print(im_norm2[:2, :2])