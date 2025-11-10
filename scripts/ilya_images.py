import os
from sys import argv
from astropy.io import fits
import numpy as np
from PIL.Image import fromarray
from tqdm import tqdm


def main():
    args = argv[1:]
    if len(args) in [0, 1]:
        raise RuntimeError('No path to Ilya fits data and/or no output path')

    ilya_fits_path = args[0]
    output_path = args[1]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for file in tqdm(os.listdir(ilya_fits_path)):
        with fits.open(f'{ilya_fits_path}/{file}') as f:
            im = np.array(f['PRIMARY'].data)
        im = im.squeeze()
        min_val = im.min()
        max_val = im.max()
        im = (im - im.min()) / (max_val - min_val) * 255
        im_uint8 = im.astype(np.uint8)
        img = fromarray(im_uint8, mode='L')
        name = file.split('.')[0]
        img.save(f'{output_path}/{name}.png', 'PNG')


if __name__ == '__main__':
    main()