#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:17:54 2023


test read UV data from astrogeo base on PV script uvfits.py


@author: lisakov
"""

import sys
sys.path.append('/home/lisakov/Programs/pypima/pypima')

import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


import numpy as np
import astropy.constants as c

from uvfits import *

uvpath= '/mnt/jet1/yyk/VLBI/RFC/images_verFeb2023/J2212+2355/'

uvfile = uvpath + 'J2212+2355_X_2018_12_20_pet_vis.fits'

print(f'Attempt to read UV file: {uvfile}')

uv = UVFits(uvfile)

#uv.u = uv.u_raw * c.c # multiply by the speed of light to get physical units of length
#uv.v = uv.v_raw * c.c

uv.u = uv.u_raw * uv.freq / 1e6 # multiply by frequency to get uv distance in wavelengths. 1e6 to convert to Megalambdas 
uv.v = uv.v_raw * uv.freq / 1e6
uv.r = np.sqrt(uv.u**2 + uv.v**2)


fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(uv.r, uv.amplitudes, 'o', label='radplot')
ax.set_ylabel('Correlated flux density [Jy]')
ax.set_xlabel(r'Baseline length projection [M$\lambda$]')
plt.savefig('lisakov.png')
plt.close(fig)