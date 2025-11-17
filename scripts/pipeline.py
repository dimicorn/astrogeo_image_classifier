#!/usr/bin/env python
import os
from sys import argv
import numpy as np
from pre.preprocess import preprocess, preprocess_lognorm
from pre.visualize import showProgress
from pre.load import fits2numpy


def main():
    args = argv[1:]
    if len(args) in [0, 1]:
        raise RuntimeError("No path to Ilya fits data and/or no output path")

    ilya_fits_path = args[0]
    output_path = args[1]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    raw_im = fits2numpy(ilya_fits_path)
    im = preprocess(raw_im)
    im_lognorm = preprocess_lognorm(raw_im)
    filename = ilya_fits_path.split("/")[-1].split(".")[0]
    np.save(f"{output_path}/{filename}", im)
    np.save(f"{output_path}/{filename}_lognorm", im_lognorm)
    showProgress(filename, raw_im, im, im_lognorm)


if __name__ == "__main__":
    main()
