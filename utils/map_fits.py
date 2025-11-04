import logging
from astropy.io import fits
import pandas as pd
import numpy as np
from utils.fits import Fits
from utils.consts import (
    PRIMARY,
    AIPS_CC,
    DATE_OBS,
    OBJECT,
    AUTHOR,
    CRPIX1,
    CRPIX2,
    CDELT1,
    CDELT2,
    NAXIS1,
    NAXIS2,
    BMAJ,
    BMIN,
    BPA,
    CRVAL3,
    FLUX,
    DELTAX,
    DELTAY,
    MAJOR_AX,
    MINOR_AX,
    POSANGLE,
    TYPE_OBJ,
    TFIELDS,
)

logger = logging.getLogger(__name__)


class MapFits(Fits):
    def __init__(self, file_name: str) -> None:
        super().__init__()
        self.file_name_w_path = file_name
        self.file_name = file_name.split("/")[-1]
        self._map_header, self._map_data = None, None
        self._cc_header, self._cc_data = None, None
        self._cc_tables = None

        with fits.open(file_name) as f:
            f.verify("fix")
            self.hdulist = f
            self._map_header = f[PRIMARY].header
            self._map_data = f[PRIMARY].data

            self.sanity_check(f)

            if len(f) == 1:
                logger.warning("%s has no CC tables", self.file_name)
            elif len(f) == 2:
                self._cc_header = f[AIPS_CC].header
                self._cc_data = f[AIPS_CC].data
                self._cc_tables = 1
            else:
                self._cc_header = f[AIPS_CC].header
                self._cc_data = f[AIPS_CC].data
                self._cc_tables = len(f) - 1
                self._cc_header, self._cc_data = [], []
                for i in range(self._cc_tables):
                    self._cc_header.append(f[i].header)
                    self._cc_data.append(f[i].data)
                logger.warning("%s has multiple CC tables", self.file_name)

        self.date = self._map_header[DATE_OBS]
        self.object = self._map_header[OBJECT]
        self.author = self._map_header[AUTHOR]
        self.freq = self.get_freq()

    def map_data(self) -> np.ndarray:
        return self._map_data

    def get_parameters(self) -> pd.DataFrame:
        """get some parameters from a header:
        CRVAL, CRPIX, FREQ, SOURCE, DATE-OBS"""
        header = self._map_header
        keys = [
            CRPIX1,
            CRPIX2,
            CDELT1,
            CDELT2,
            NAXIS1,
            NAXIS2,
            BMAJ,
            BMIN,
            BPA,
            OBJECT,
            DATE_OBS,
            CRVAL3,
        ]
        params = {key: np.array([header[key]]) for key in keys}
        params[CDELT1] *= 3.6e6
        params[CDELT2] *= 3.6e6
        return pd.DataFrame(params)

    def map_noise(self, data: np.ndarray, k: float = 0.1) -> float:
        # TODO: check indexes
        # borders
        b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
        b3, b4 = int((1 - k) * data.shape[0]), int((1 - k) * data.shape[1])
        upper_left = np.std(data[:b1, :b2])
        upper_right = np.std(data[b3:, :b2])
        down_left = np.std(data[:b1, b4:])
        down_right = np.std(data[b3:, b4:])
        noise = np.median([upper_left, upper_right, down_left, down_right])
        return noise

    def get_sql_params(self) -> tuple:
        """
        object_name, obs_date, freq, obs_author, file_name,
        map_max, mapc_x, mapc_y, map_max_x, map_max_y,
        map_max_x_mas, map_max_y_mas, noise_level,
        map_size_x, map_size_y, pixel_size_x, pixel_size_y,
        b_maj, b_min, b_pa, cc_tables
        """
        cc_tables = self._cc_tables

        pixel_size_x = self.header_key_check(CDELT1) * 3.6e6
        pixel_size_y = self.header_key_check(CDELT2) * 3.6e6

        mapc_x, mapc_y, map_max_x, map_max_y, _, map_max, noise_level = (
            self.get_quality_params()
        )

        # строчки и столбцы
        map_max_x_mas = map_max_x * pixel_size_x
        map_max_y_mas = map_max_y * pixel_size_y
        noise = (
            map_max,
            mapc_x,
            mapc_y,
            float(map_max_x),
            float(map_max_y),
            float(map_max_x_mas),
            float(map_max_y_mas),
            float(noise_level),
        )

        map_size_x = self.header_key_check(NAXIS1)
        map_size_y = self.header_key_check(NAXIS2)
        bmaj = self.header_key_check(BMAJ)
        bmin = self.header_key_check(BMIN)
        bpa = self.header_key_check(BPA)

        map_params = (
            map_size_x,
            map_size_y,
            pixel_size_x,
            pixel_size_y,
            bmaj,
            bmin,
            bpa,
            cc_tables,
        )

        data = self.header_data() + noise + map_params
        return data

    def get_quality_params(self) -> tuple[float]:
        mapc_x = self.header_key_check(CRPIX1)
        mapc_y = self.header_key_check(CRPIX2)
        map_data = self.map_data().squeeze()
        index = np.argmax(map_data)
        coords = np.unravel_index(index, map_data.shape)
        map_max_y, map_max_x = coords[0], coords[1]
        header_data = self.header_data()
        author = header_data[3]
        map_max = self.header_key_check("DATAMAX")
        # map_max = np.max(map_data)
        noise_level = self.map_noise(map_data)
        return (
            mapc_x,
            mapc_y,
            map_max_x,
            map_max_y,
            author,
            map_max,
            noise_level,
        )

    def get_models(self) -> pd.DataFrame:
        models = pd.DataFrame()
        keys = [FLUX, DELTAX, DELTAY, MAJOR_AX, MINOR_AX, POSANGLE, TYPE_OBJ]
        new_keys = [FLUX, DELTAX, DELTAY, "MAJOR_AX", "MINOR_AX", POSANGLE, "TYPE_OBJ"]

        # If multiple CC tables, using the first one
        field_num = self._cc_header[TFIELDS]
        for _, key, new_key in zip(range(field_num), keys, new_keys):
            models[new_key] = self._cc_data[key].tolist()

        if not (field_num == 7 or field_num == 3):
            logger.error("Wrong number of columns in CC table %s", self.file_name)
        return models
