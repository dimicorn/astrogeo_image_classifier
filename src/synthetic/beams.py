import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import unit_impulse, convolve2d
from skimage.draw import line


def sqr(x: float) -> float: return x * x

class Beams(object):
    rgb = 255
    def __init__(self, shape: tuple) -> None:
        self.shape = shape
    
    def point_beam(self, point: tuple = (0, 0)) -> np.array:
        return unit_impulse(self.shape, point) * self.rgb
    
    def two_points_beam(
            self, d: int, alpha: float, point: tuple = (0, 0)
        ) -> np.array:
        x0, y0 = point
        point2 = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        return self.point_beam(point) + self.point_beam(point2)
    
    def gauss_beam(
            self, b_maj: int, b_min: int, b_pa: float, point=(0, 0)
        ) -> np.array:
        x0, y0 = point
        a = (sqr(np.cos(b_pa)) / (2 * sqr(b_maj)) + 
             sqr(np.sin(b_pa)) / (2 * sqr(b_min)))
        b = (-np.sin(2 * b_pa) / (4 * sqr(b_maj)) + 
             np.sin(2 * b_pa) / (4 * sqr(b_min)))
        c = (sqr(np.sin(b_pa)) / (2 * sqr(b_maj)) + 
             sqr(np.cos(b_pa)) / (2 * sqr(b_min)))
        x, y = np.meshgrid(np.linspace(-self.rgb-1, self.rgb, self.shape[0]), 
                           np.linspace(-self.rgb-1, self.rgb, self.shape[1]))
        e = np.exp(-(a * sqr(x-x0) + 2 * b * (x-x0) * (y-y0) + c * sqr(y-y0)))
        return e * self.rgb
    
    def two_gauss_beam(
            self, b_maj1: int, b_min1: int, b_pa1: float, point: tuple, 
            b_maj2: int, b_min2: int, b_pa2: float, d: int, alpha: float
        ) -> np.array:
        x0, y0 = point
        point2 = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        gauss1 = self.gauss_beam(b_maj1, b_min1, b_pa1, point)
        gauss2 = self.gauss_beam(b_maj2, b_min2, b_pa2, point2)
        return gauss1 + gauss2
        
    def gauss_w_jet_beam(
            self, b_maj: int, b_min: int, b_pa: float,
            d: int, alpha: float, point=(0, 0)
        ) -> np.array:
        x0, y0 = self.shape[0]//2 + point[0], self.shape[1]//2 + point[1]
        gauss_w_jet = self.gauss_beam(b_maj, b_min, b_pa, point)
        jet = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        rr, cc = line(x0, y0, jet[0], jet[1])
        gauss_w_jet[rr, cc] = self.rgb
        return gauss_w_jet
    
    def gauss_w_two_jets_beam(
            self, b_maj: int, b_min: int, b_pa: float,
            d: int, alpha: float, point=(0, 0)
        ) -> np.array:
        x0, y0 = self.shape[0]//2 + point[0], self.shape[1]//2 + point[1]
        gauss_w_jets = self.gauss_beam(b_maj, b_min, b_pa, point)
        jet1 = (int(x0 - d * np.sin(alpha)), int(y0 - d * np.cos(alpha)))
        jet2 = (int(x0 + d * np.sin(alpha)), int(y0 + d * np.cos(alpha)))
        rr, cc = line(jet1[0], jet1[1], jet2[0], jet2[1])
        gauss_w_jets[rr, cc] = self.rgb
        return gauss_w_jets
    
    def gauss_w_spiral_beam(
            self, b_maj: int, b_min: int, b_pa: float,
            v: float, c: float, w: float, phi: float, point=(0, 0)
        ) -> np.array:
        x0, y0 = self.shape[0]//2 + point[0], self.shape[1]//2 + point[1]
        gauss = self.gauss_beam(b_maj, b_min, b_pa, point)
        t = np.linspace(0, 63, self.rgb)
        x = np.array((v * t + c) * np.cos(w * t + phi) + x0).astype(int)
        y = np.array((v * t + c) * np.sin(w * t + phi) + y0).astype(int)
        gauss[x, y] = self.rgb
        return gauss

    def draw_beam(self, beam: np.array, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(beam, interpolation='none', origin='lower')
        ax.set_xlabel('x [pixels]')
        ax.set_ylabel('y [pixels]')
        fig.colorbar(im)
        fig.savefig(f'test/{filename}', dpi=500)
        plt.close(fig)
    
    def conv(self, model: np.array, kernel: np.array) -> np.array:
        return convolve2d(model, kernel)