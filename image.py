from uvfits import UVFits
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from astropy.io import fits


class Image:
    def __init__(self):
        with open('config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_path = config['path']

        self.object_uv_data = []
        self.object_map_data = []

    def get_objects(self) -> list[str]:
        object_dirs = os.listdir(path=self.data_path)
        return object_dirs

    def get_date(self, data_file : str) -> str:
        date = '-'.join(data_file.split('_')[2:5])
        return date

    def draw_uv(self, obj: str) -> None:
        if len(self.object_uv_data) == 0:
            for file in os.listdir(f'{self.data_path}/{obj}'):
                if file[-8:] == 'vis.fits':
                    self.object_uv_data.append(file)

        for data_file in self.object_uv_data:
            data = UVFits(f'{self.data_path}/{obj}/{data_file}')
            
            # freq = [i['if_freq'] for i in data.freq_table]
            # print(freq)
            # print(np.mean(freq) * 1e-9)

            # data.print_info()
            # data.print_uv()

            uu, vv = data.get_uv()
            uu = np.array(uu)
            vv = np.array(vv)
            # print(uu, vv)
            plt.scatter(uu * 1e-6, vv * 1e-6, marker='.', color='blue')
            # symmetrical points
            plt.scatter(uu * -1e-6, vv * -1e-6, marker='.', color='blue')
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)
            plt.xlabel(r'U Baseline projection (M$\lambda$)')
            plt.ylabel(r'V Baseline projection (M$\lambda$)')

            plt.title(obj, loc='center')
            date = self.get_date(data_file)
            plt.title(date, loc='left')

            # Should be redone
            if data_file.split('_')[1] == 'C':
                freq = "4.3 GHz"
            
            plt.title(freq, loc='right')
            plt.savefig(f'uv_plots/{data_file}.png', dpi=500)
            # plt.show()

    def draw_map(self, obj: str) -> None:
        if len(self.object_map_data) == 0:
            for file in os.listdir(f'{self.data_path}/{obj}'):
                if file[-8:] == 'map.fits':
                    self.object_map_data.append(file)

        for data_file in self.object_map_data:
            # print(data_file)
            data = fits.open(f'{self.data_path}/{obj}/{data_file}')
            #print(fits.info(f'{self.data_path}/{obj}/{data_file}'))
            #print(data[1].header[:5])
            # print(data[0].data.shape)
            map2d = data[0].data.squeeze()
            plt.imshow(map2d, cmap='magma')
            plt.savefig(f'maps/{data_file}.png', dpi=500)  
