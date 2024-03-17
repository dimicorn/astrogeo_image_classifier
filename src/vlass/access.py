from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.cadc import Cadc
import matplotlib.pyplot as plt
import numpy as np
import warnings
from bs4 import XMLParsedAsHTMLWarning


warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
cadc = Cadc()
coords = SkyCoord(10, 20, unit='deg')
radius = 0.01 * u.deg
readable_objs = cadc.get_images_async(coords, radius, collection='VLASS')

for ind, obj in enumerate(readable_objs):
    hdu = obj.get_fits()
    image = hdu[0].data
    image = image.squeeze()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(image, cmap='magma', origin='lower')
    fig.colorbar(im)
    fig.savefig(f'src/vlass/test/{ind}.png', dpi=500)
    plt.close(fig)