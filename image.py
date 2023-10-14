import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import numpy as np
from consts import *
from fits import UVFits, MapFits, FitsError

# TODO: use amplitude in LOF

class Image(UVFits, MapFits):
    def __init__(self, file_name: str) -> None:
        if file_name[-8:] == VIS_FITS:
            UVFits.__init__(self, file_name)
        elif file_name[-8:] == MAP_FITS:
            MapFits.__init__(self, file_name)
        else:
            raise FitsError('Wrong extension of file', self.file_name)

    def draw_uv(self) -> None:
        # checking if directory for UV plots exists, if not creates it
        if not os.path.exists(UV_DIR):
            os.makedirs(UV_DIR)
        
        X = self.uv_data()
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0] * 1e-6, X[:, 1] * 1e-6, marker=DOT, color=COLOR)
        # ax.scatter(X[:, 0] * -1e-6, X[:, 1] * -1e-6, marker=DOT, color=COLOR) # symmetrical points
        ax.set_xlabel(r'U Baseline projection (M$\lambda$)')
        ax.set_ylabel(r'V Baseline projection (M$\lambda$)')

        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)
        uv_plot_name = self.file_name.split('/')[-1][:-9]
        fig.savefig(f'{UV_DIR}/{uv_plot_name}.png', dpi=500)
        plt.close(fig)
    
    def _update_legend_marker_size(self, handle, orig) -> None:
        'Customize size of the legend marker'
        handle.update_from(orig)
        handle.set_sizes([20])
    
    def draw_uv_lof(self) -> None:
        if not os.path.exists(LOF_DIR):
            os.makedirs(LOF_DIR)
        
        X = self.uv_data()
        inl, outl, scores = self.uv_outliers()

        fig, ax = plt.subplots()
        ax.scatter(inl[:, 0] * 1e-6, inl[:, 1] * 1e-6, marker=DOT, color='b', label='Inliers')
        ax.scatter(outl[:, 0] * 1e-6, outl[:, 1] * 1e-6, marker=DOT, color='r', label='Outliers')
        # plot circles with radius proportional to the outlier scores
        radius = abs(np.mean(scores) - scores) / np.std(scores)
        radus = (np.max(scores) - scores) / (np.max(scores) - np.min(scores))
        scatter = ax.scatter(X[:, 0] * 1e-6, X[:, 1] * 1e-6,
            s=10 * radius,
            edgecolors='g',
            facecolors='none',
            label=r'$\frac{|\overline{x} - x_i|}{\sigma}$'
            # label=r'$\frac{x_{max} - x_i}{x_{max} - x_{min}}$'
        )
        ax.axis('tight')
        ax.legend(
            handler_map={scatter: HandlerPathCollection(update_func=self._update_legend_marker_size)}
        )
        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)
        lof_plot_name = self.file_name.split('/')[-1][:-9]
        plt.savefig(f'{LOF_DIR}/{lof_plot_name}.png', dpi=500)
        plt.close(fig)

    def draw_map(self) -> None:
        # checking if directory for maps exists, if not creates it
        if not os.path.exists(MAP_DIR):
            os.makedirs(MAP_DIR)
        
        data = self.map_data()
        map2d = data.squeeze()

        fig, ax = plt.subplots()
        ax.imshow(map2d, cmap=CMAP)

        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)

        map_plot_name = self.file_name.split('/')[-1][:-9]
        fig.savefig(f'{MAP_DIR}/{map_plot_name}.png', dpi=500)
        plt.close(fig)
    
    """
    def freq(self, obj: str) -> None:
        data_file = self.obj_uv_files(obj)[0]
        uv = fits.open(f'{self.data_path}/{obj}/{data_file}')
        # fits.info(f'{self.data_path}/{obj}/{data_file}')

        # print(uv[2].header['FREQ'] * 1e-9) # <- frequency
        # print(uv[2].header['EXTVER']) # <- number of subarrays

        '''
        for i in range(5):
            print(uv[1].header[f'TTYPE{i+1}'], uv[1].header[f'TUNIT{i+1}'])
        for i in range(12):
            print(uv[2].header[f'TTYPE{i+1}'], uv[2].header[f'TUNIT{i+1}'])
        '''
        '''
        for i in uv[1].header:
            print(i, uv[1].header[i])
        for i in uv[2].header:
            print(i, uv[2].header[i])
        '''

        # print(list(uv[0].header.keys()))
        # print(uv[0].data)
        # print(uv[2].data.field(0))
        # print(inspect.getmembers(uv[0].data))
        # print(uv[2].data['ANNAME'])
        # print(uv[2].data['STABXYZ'])
        # print(uv[2].data['ORBPARM'])
        '''
        keys = ['UU', 'VV', 'WW', 'BASELINE', 'DATE', '_DATE', 'INTTIM', 'DATA']
        for i in keys:
            print(i, uv[0].data[i])
        '''
        # uv.info()
        '''
        Primary table -- data
        ['UU', 'VV', 'WW', 'BASELINE', 'DATE', '_DATE', 'INTTIM', 'DATA']
        AN table -- data
        ['ANNAME', 'STABXYZ', 'ORBPARM', 'NOSTA', 'MNTSTA',\
        'STAXOF', 'POLTYA', 'POLAA', 'POLCALA', 'POLTYB',\
        'POLAB', 'POLCALB']

        FQ table -- data
        ['FRQSEL', 'IF FREQ', 'CH WIDTH', 'TOTAL BANDWIDTH',\
        'SIDEBAND']
        '''
    """
