from astropy.io import fits
import numpy as np


def fits2numpy(path2file: str) -> np.ndarray:
    with fits.open(path2file) as f:
        raw_image = np.array(f['PRIMARY'].data, dtype=np.float32)
    # removing excessive dimensions, leaving only height & width
    return raw_image.squeeze()