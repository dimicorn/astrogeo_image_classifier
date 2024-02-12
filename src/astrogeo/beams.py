import os
import numpy as np
import pandas as pd
from matplotlib.image import imsave
from scipy.signal import unit_impulse
from astropy.convolution import convolve_fft
from skimage.draw import line


def sqr(x: float) -> float: return x * x

class Beams(object):
    rgb = 255
    def __init__(self, shape: tuple = (512, 512)) -> None:
        self.shape = shape
        self.kernels = None
    
    def point_beam(self, point: tuple = (0, 0)) -> np.array:
        coords = (point[0] + self.shape[0]//2, point[1] + self.shape[1]//2)
        return unit_impulse(self.shape, coords) * self.rgb
    
    def two_points_beam(
            self, d: int, alpha: float, point: tuple = (0, 0)
        ) -> np.array:
        x0, y0 = point
        point2 = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        two_points = self.point_beam(point) + self.point_beam(point2)
        return two_points / np.max(two_points) * self.rgb
    
    def gauss_beam(
            self, b_maj: int, b_min: int, b_pa: float, 
            shape: tuple = None, point: tuple = (0, 0)
        ) -> np.array:
        if shape is None:
            shape = self.shape
        x0, y0 = point
        a = (sqr(np.cos(b_pa)) / (2 * sqr(b_maj)) + 
             sqr(np.sin(b_pa)) / (2 * sqr(b_min)))
        b = (-np.sin(2 * b_pa) / (4 * sqr(b_maj)) + 
             np.sin(2 * b_pa) / (4 * sqr(b_min)))
        c = (sqr(np.sin(b_pa)) / (2 * sqr(b_maj)) + 
             sqr(np.cos(b_pa)) / (2 * sqr(b_min)))
        x, y = np.meshgrid(np.linspace(-shape[0]//2, shape[0]//2-1, shape[0]), 
                           np.linspace(-shape[1]//2, shape[1]//2-1, shape[1]))
        e = np.exp(-(a * sqr(x-x0) + 2 * b * (x-x0) * (y-y0) + c * sqr(y-y0)))
        return e * self.rgb
    
    def _point_beam(self, point: tuple = (0, 0)) -> np.array:
        return self.gauss_beam(1, 1, 0, point=point)

    def two_gauss_beam(
            self, b_maj1: int, b_min1: int, b_pa1: float, 
            b_maj2: int, b_min2: int, b_pa2: float,
            d: int, alpha: float, point: tuple = (0, 0)
        ) -> np.array:
        x0, y0 = point
        point2 = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        gauss1 = self.gauss_beam(b_maj1, b_min1, b_pa1, point=point)
        gauss2 = self.gauss_beam(b_maj2, b_min2, b_pa2, point=point2)
        two_gauss = gauss1 + gauss2
        return two_gauss / np.max(two_gauss) * self.rgb
    
    def _two_points_beam(
            self, d: int, alpha: float, point: tuple = (0, 0)
        ) -> np.array:
        return self.two_gauss_beam(1, 1, 0, 1, 1, 0, d, alpha, point=point)
    
    def gauss_w_jet_beam(
            self, b_maj: int, b_min: int, b_pa: float,
            d: int, alpha: float, point: tuple = (0, 0)
        ) -> np.array:
        x0, y0 = self.shape[0]//2 + point[0], self.shape[1]//2 + point[1]
        gauss_w_jet = self.gauss_beam(b_maj, b_min, b_pa, point=point)
        jet = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        rr, cc = line(x0, y0, jet[0], jet[1])
        gauss_w_jet[rr, cc] = self.rgb
        return gauss_w_jet
    
    def gauss_w_two_jets_beam(
            self, b_maj: int, b_min: int, b_pa: float,
            d: int, alpha: float, point: tuple = (0, 0)
        ) -> np.array:
        x0, y0 = self.shape[0]//2 + point[0], self.shape[1]//2 + point[1]
        gauss_w_jets = self.gauss_beam(b_maj, b_min, b_pa, point=point)
        jet1 = (int(x0 - d * np.sin(alpha)), int(y0 - d * np.cos(alpha)))
        jet2 = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        rr, cc = line(jet1[0], jet1[1], jet2[0], jet2[1])
        gauss_w_jets[rr, cc] = self.rgb
        return gauss_w_jets
    
    def gauss_w_spiral_beam(
            self, b_maj: int, b_min: int, b_pa: float,
            v: float, c: float, w: float, phi: float, point: tuple = (0, 0)
        ) -> np.array:
        x0, y0 = self.shape[0]//2 + point[0], self.shape[1]//2 + point[1]
        gauss = self.gauss_beam(b_maj, b_min, b_pa, point=point)
        t = np.linspace(0, self.shape[0]//8-1, self.rgb)
        x = np.array((v * t + c) * np.cos(w * t + phi) + x0).astype(int)
        y = np.array((v * t + c) * np.sin(w * t + phi) + y0).astype(int)
        gauss[x, y] = self.rgb
        return gauss

    def draw_beam(self, beam: np.array, filename: str, path: str = None) -> None:
        if path is None:
            path = 'src/astrogeo/test'
        if not os.path.exists(path):
            os.makedirs(path)
        
        imsave(f'{path}/{filename}.png', np.log10(beam + 1), origin='lower')
        # fig, ax = plt.subplots(figsize=(10, 8))
        # beam += 1
        # im = ax.imshow(np.log10(beam), interpolation='none', origin='lower')
        # ax.set_xlabel('x [pixels]')
        # ax.set_ylabel('y [pixels]')
        # fig.colorbar(im)
        # fig.savefig(f'{path}/{filename}', dpi=500)
        # plt.close(fig)
    
    def conv(self, model: np.array, kernel: np.array) -> np.array:
        c = convolve_fft(model, kernel)
        return c / np.max(c) * self.rgb
    
    def test_beams(self) -> list:
        d, alpha = self.shape[0]//4, np.pi/4
        b_maj, b_min, b_pa = 15, 10, np.pi/4
        b_maj2, b_min2, b_pa2 = 8, 8, 0
        v, c, w = 1, 0, 0.05
        beams = [
            self.point_beam(), self.two_points_beam(d, alpha),
            self.gauss_beam(b_maj, b_min, b_pa),
            self.two_gauss_beam(
                b_maj, b_min, b_pa, b_maj2, b_min2, b_pa2, d, alpha
            ),
            self.gauss_w_jet_beam(b_maj, b_min, b_pa, d, alpha),
            self.gauss_w_two_jets_beam(b_maj, b_min, b_pa, d, alpha),
            self.gauss_w_spiral_beam(b_maj, b_min, b_pa, v, c, w, alpha)
        ]
        return beams
    
    def draw_test_beams(self) -> None:
        beams = self.test_beams()
        file_names = ['point_beam.png', 'two_point_beam.png',
                      'gauss_beam.png',  'two_gauss_beam.png', 
                      'gauss_w_jet_beam.png', 'gauss_w_two_jets_beam.png',
                      'gauss_w_spiral_beam.png']
        for beam, name in zip(beams, file_names):
            self.draw_beam(beam, name)
    
    def kernel(
            self, b_maj: int, b_min: int, b_pa: float,
            shape: tuple = None, point: tuple = (0, 0)
        ) -> np.array:
        if shape is None:
            shape = (self.shape[0]//8, self.shape[1]//8)
        return self.gauss_beam(b_maj, b_min, b_pa, shape=shape, point=point)
    
    def get_kernels(self, cluster_means: pd.DataFrame) -> list:
        if self.kernels is not None:
            return self.kernels
        rows = cluster_means.shape[0]
        alpha, ratio = np.sqrt(np.log10(2)), 6 # wtf is this 
        self.kernels = []
        for row in range(rows):
            datum = cluster_means.loc[row].to_list()[1:4]
            sigma = max(datum[0], datum[1]) / alpha
            sh = np.ceil(ratio * sigma).astype(int)
            kernel_size = (sh, sh)
            k = self.kernel(*datum, shape=kernel_size)
            self.kernels.append(k)
        return self.kernels
    
    def draw_kernels(self, cluster_means: pd.DataFrame) -> None:
        kernels = self.get_kernels(cluster_means)
        for ind, kernel in enumerate(kernels):
            self.draw_beam(kernel, f'kernel_{ind}')
    
    def conv_beams(self, cluster_means: pd.DataFrame, path: str) -> None:
        kernels = self.get_kernels(cluster_means)
        models = self.test_beams()
        for m_i, model in enumerate(models):
            if not os.path.exists(f'{path}/{m_i}'):
                os.makedirs(f'{path}/{m_i}')
            for k_i, kernel in enumerate(kernels):
                self.draw_beam(self.conv(model, kernel), str(k_i), path=f'{path}/{m_i}')