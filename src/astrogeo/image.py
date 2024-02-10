import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.neighbors import LocalOutlierFactor as lof
import numpy as np
from astrogeo.consts import *
from astrogeo.fits import UVFits, MapFits, FitsError


class Image(UVFits, MapFits):
    def __init__(self, file_name: str) -> None:
        if file_name[-8:] == VIS_FITS:
            UVFits.__init__(self, file_name)
        elif file_name[-8:] == MAP_FITS:
            MapFits.__init__(self, file_name)
        else:
            raise FitsError('Wrong extension of file', self.file_name)
        self.test_dir = 'src/astrogeo/test'
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def draw_uv(self) -> None:
        # checking if directory for UV plots exists, if not creates it
        if not os.path.exists(UV_DIR):
            os.makedirs(UV_DIR)
        
        X = self.uv_data()
        fig, ax = plt.subplots()
        ax.scatter(X[0] * 1e-6, X[1] * 1e-6, marker=DOT, color=COLOR)
        ax.set_xlabel(r'U Baseline projection (M$\lambda$)')
        ax.set_ylabel(r'V Baseline projection (M$\lambda$)')

        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)
        uv_plot_name = self.file_name[:-9]
        fig.savefig(f'{UV_DIR}/{uv_plot_name}.png', dpi=500)
        plt.close(fig)
    
    def _update_legend_marker_size(self, handle, orig) -> None:
        '''Customize size of the legend marker'''
        handle.update_from(orig)
        handle.set_sizes([20])
    
    def draw_uv_lof(self) -> None:
        if not os.path.exists(LOF_DIR):
            os.makedirs(LOF_DIR)

        X = self.uv_data()[0:2]
        clf = lof(n_neighbors=75, contamination=0.01)
        y_pred = clf.fit_predict(X.T)
        X_scores = clf.negative_outlier_factor_
        scores = X_scores
        inl, outl = [], []
        for i, val in enumerate(y_pred):
            if val == -1:
                outl.append([X[0][i], X[1][i]])
            elif val == 1:
                inl.append([X[0][i], X[1][i]])
        
        inl, outl = np.array(inl), np.array(outl)

        fig, ax = plt.subplots()
        ax.scatter(
            inl[:, 0] * 1e-6,
            inl[:, 1] * 1e-6,
            marker=DOT, color='b', label='Inliers'
        )
        ax.scatter(
            outl[:, 0] * 1e-6,
            outl[:, 1] * 1e-6,
            marker=DOT, color='r', label='Outliers'
        )
        # plot circles with radius proportional to the outlier scores
        # radius = abs(np.mean(scores) - scores) / np.std(scores)
        radius = (np.max(scores) - scores) / (np.max(scores) - np.min(scores))
        scatter = ax.scatter(X[0] * 1e-6, X[1] * 1e-6,
            s=10 * radius,
            edgecolors='g',
            facecolors='none',
            label=r'$\frac{|\overline{x} - x_i|}{\sigma}$'
            # label=r'$\frac{x_{max} - x_i}{x_{max} - x_{min}}$'
        )
        ax.axis('tight')
        ax.legend(
            handler_map={
                scatter: HandlerPathCollection(
                    update_func=self._update_legend_marker_size
                )
            }
        )
        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)
        ax.set_xlabel(r'U Baseline projection (M$\lambda$)')
        ax.set_ylabel(r'V Baseline projection (M$\lambda$)')
        lof_plot_name = self.file_name[:-9]
        plt.savefig(f'{self.test_dir}/test_lof_2d.png', dpi=500)
        plt.close(fig)
    
    def draw_uv_3d(self) -> None:
        X = self.uv_data()[0:3]
        clf = lof(n_neighbors=75, contamination=0.01)
        y_pred = clf.fit_predict(X.T)
        X_scores = clf.negative_outlier_factor_
        scores = X_scores
        inl, outl = [], []
        for i, val in enumerate(y_pred):
            if val == -1:
                outl.append([X[0][i], X[1][i], X[2][i]])
            elif val == 1:
                inl.append([X[0][i], X[1][i], X[2][i]])
        
        inl, outl = np.array(inl), np.array(outl)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            inl[:, 0] * 1e-6,
            inl[:, 1] * 1e-6,
            inl[:, 2],
            marker=DOT, color='b', label='Inliers'
        )
        ax.scatter(
            outl[:, 0] * 1e-6,
            outl[:, 1] * 1e-6,
            outl[:, 2],
            marker=DOT, color='r', label='Outliers'
        )
        # plot circles with radius proportional to the outlier scores
        # radius = abs(np.mean(scores) - scores) / np.std(scores)
        radius = (np.max(scores) - scores) / (np.max(scores) - np.min(scores))
        scatter = ax.scatter(X[0] * 1e-6, X[1] * 1e-6, X[2],
            s=10 * radius,
            edgecolors='g',
            facecolors='none',
            # label=r'$\frac{|\overline{x} - x_i|}{\sigma}$'
            label=r'$\frac{x_{max} - x_i}{x_{max} - x_{min}}$'
        )
        ax.axis('tight')
        ax.legend(
            handler_map={
                scatter: HandlerPathCollection(
                    update_func=self._update_legend_marker_size
                )
            }
        )
        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)
        ax.set_xlabel(r'U Baseline projection (M$\lambda$)')
        ax.set_ylabel(r'V Baseline projection (M$\lambda$)')
        ax.set_zlabel('Amplitude, Jy')
        lof_plot_name = self.file_name[:-9]
        plt.savefig(f'{self.test_dir}/test_lof_3d.png', dpi=500)
        plt.close(fig)
    
    def draw_phase_radius(self) -> None:
        X = self.uv_data()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(
            np.sqrt(np.square(X[0]) + np.square(X[1])) * 1e-6,
            X[3],
            marker='.'
        )
        ax.set_xlabel(r'$R = \sqrt{U^2 + V^2}$ Baseline radius (M$\lambda$)')
        ax.set_ylabel(r'$\varphi$ phase, radians')
        ax.set_title(self.object, loc=CENTER)
        plt.savefig(f'{self.test_dir}/test_phase_radius.png', dpi=500)
        plt.close(fig)
    
    def draw_ampl_radius(self) -> None:
        X = self.uv_data()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(
            np.sqrt(np.square(X[0]) + np.square(X[1])) * 1e-6,
            X[2],
            marker='.'
        )
        ax.set_title(self.object, loc=CENTER)
        ax.set_xlabel(r'$R = \sqrt{U^2 + V^2}$ Baseline radius (M$\lambda$)')
        ax.set_ylabel('Amplitude, Jy')
        plt.savefig(f'{self.test_dir}/test_ampl_radius.png', dpi=500)
        plt.close(fig)

    def draw_map(self) -> None:
        # checking if directory for maps exists, if not creates it
        if not os.path.exists(MAP_DIR):
            os.makedirs(MAP_DIR)

        data = self.map_data()
        map2d = data.squeeze()
        # noise = self.map_noise(map2d)
        # map2d = np.where(map2d > 5 * noise, map2d, 0)
        fig, ax = plt.subplots(figsize=(10, 8))
        # ax.imshow(np.log10(map2d), cmap=CMAP, origin='lower')
        ax.imshow(map2d, cmap=CMAP, origin='lower')
        ax.set_title(self.object, loc=CENTER)
        ax.set_title(self.date, loc=LEFT)
        ax.set_title(f'{self.freq * 1e-9:.1f} GHz', loc=RIGHT)

        map_plot_name = self.file_name[:-9]
        # {MAP_DIR}/{map_plot_name}
        fig.savefig(f'{MAP_DIR}/{map_plot_name}.png', dpi=500)
        plt.close(fig)