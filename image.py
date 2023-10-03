import os
import yaml
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from uvfits import UVFits
from consts import *


class Image(object):
    def __init__(self) -> None:
        with open('config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_path = config['path']

    def get_objects(self) -> list[str]:
        return os.listdir(path=self.data_path)

    def obj_uv_files(self, obj) -> list[str]:
        return [file for file in os.listdir(f'{self.data_path}/{obj}') 
                if file[-8:] == vis_fits]
    
    def obj_map_files(self, obj) -> list[str]:
        return [file for file in os.listdir(f'{self.data_path}/{obj}') 
                if file[-8:] == map_fits]

    def get_date(self, data_file: str) -> str:
        return '-'.join(data_file.split('_')[2:5])

    def draw_uv(self, obj: str) -> None:
        if not os.path.exists(uv_dir):
            os.makedirs(uv_dir)

        for data_file in self.obj_uv_files(obj):
            data = UVFits(f'{self.data_path}/{obj}/{data_file}')
            uu, vv = np.array(data.get_uv())

            fig, ax = plt.subplots()
            ax.scatter(uu * 1e-6, vv * 1e-6, marker=dot, color=blue)
            ax.scatter(uu * -1e-6, vv * -1e-6, marker=dot, color=blue) # symmetrical points
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_xlabel(r'U Baseline projection (M$\lambda$)')
            ax.set_ylabel(r'V Baseline projection (M$\lambda$)')

            # Should be redone
            freq = "4.3 GHz"

            ax.set_title(obj, loc=center)
            ax.set_title(self.get_date(data_file), loc=left)
            ax.set_title(freq, loc=right)

            fig.savefig(f'{uv_dir}/{data_file}.png', dpi=500)
            plt.close(fig)

    def draw_map(self, obj: str) -> None:
        # checking if directory for maps exists, if not creates it
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        
        for data_file in self.obj_map_files(obj):
            with fits.open(f'{self.data_path}/{obj}/{data_file}') as data:
                fig, ax = plt.subplots()
                map2d = data[0].data.squeeze()
                ax.imshow(map2d, cmap='magma')

                # Should be redone
                freq = "4.3 GHz"

                ax.set_title(obj, loc=center)
                ax.set_title(self.get_date(data_file), loc=left)
                ax.set_title(freq, loc=right)

                fig.savefig(f'{map_dir}/{data_file}.png', dpi=500)
                plt.close(fig)

    def get_parameters(self, obj: str) -> pd.DataFrame:
        ''' get some parameters from a header: CRVAL, CRPIX, FREQ, SOURCE, DATE-OBS'''
        object_map_data = self.obj_map_files(obj)
        
        with fits.open(f'{self.data_path}/{obj}/{object_map_data[0]}') as fits_obj:
            header = fits_obj[0].header
        
        param = pd.DataFrame(columns = ['racenpix', 'deccenpix', # central pixel coords in px
                                        'rapixsize', 'decpixsize', # pixel size in degrees
                                        'ramapsize', 'decmapsize', # map size in px
                                        'bmaj', 'bmin', 'bpa', # degrees, to be checked
                                        'source', 'dateobs',
                                        'frequency', # Hz
                                        'masperpix', 'masperpixx', 'masperpixy'], # pixel size in mas
                            data = [[header['CRPIX1'], header['CRPIX2'],
                                header['CDELT1'], header['CDELT2'], 
                                header['NAXIS1'], header['NAXIS2'],
                                header['BMAJ'], header['BMIN'], header['BPA'],
                                header['OBJECT'], header['DATE-OBS'],
                                header['CRVAL3'], 
                                np.abs(header['CDELT1']) * 3.6e6, 
                                header['CDELT1'] * 3.6e6, header['CDELT2'] * 3.6e6]])
        # MASperPIX = np.abs(self.rm[0].header['CDELT1']*3.6e6)
        return param