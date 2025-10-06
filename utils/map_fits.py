from astropy.io import fits
import pandas as pd
import numpy as np
from utils.fits import FitsError, Fits
from utils.consts import *


# TODO: Add logging as in uv_fits
class MapFits(Fits):
    def __init__(self, file_name) -> None:
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

            self.sanityCheck(f)

            if len(f) == 1:
                print(f"Caution: {self.file_name} has no CC tables")
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
                print(f"Caution: {self.file_name} has multiple CC tables")

        self.date = self._map_header[DATE_OBS]
        self.object = self._map_header[OBJECT]
        self.author = self._map_header[AUTHOR]
        self.freq = self.getFreq()

    def mapData(self):
        return self._map_data

    def getParameters(self) -> pd.DataFrame:
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

    def mapNoise(self, data, k=0.1) -> float:
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

    def getSQLParams(self) -> tuple:
        """
        object_name, obs_date, freq, obs_author, file_name,
        map_max, mapc_x, mapc_y, map_max_x, map_max_y,
        map_max_x_mas, map_max_y_mas, noise_level,
        map_size_x, map_size_y, pixel_size_x, pixel_size_y,
        b_maj, b_min, b_pa, cc_tables,
        map_quality, comment
        """
        header_data = self.headerData()
        header, cc_tables = self._map_header, self._cc_tables

        map_data = self.mapData().squeeze()
        map_max = self.headerKeyCheck("DATAMAX")
        # map_max = np.max(map_data)
        mapc_x = self.headerKeyCheck(CRPIX1)
        mapc_y = self.headerKeyCheck(CRPIX2)
        pixel_size_x = self.headerKeyCheck(CDELT1) * 3.6e6
        pixel_size_y = self.headerKeyCheck(CDELT2) * 3.6e6
        ind = np.argmax(map_data)

        # строчки и столбцы
        map_max_y, map_max_x = np.unravel_index(ind, map_data.shape)
        noise_level = self.mapNoise(map_data)
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

        map_size_x = self.headerKeyCheck(NAXIS1)
        map_size_y = self.headerKeyCheck(NAXIS2)
        bmaj = self.headerKeyCheck(BMAJ)
        bmin = self.headerKeyCheck(BMIN)
        bpa = self.headerKeyCheck(BPA)

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

        map_quality, comment = self.qualityComment(
            mapc_x, mapc_y, map_max_x, map_max_y, header_data[3], map_max, noise_level
        )
        data = header_data + noise + map_params + (map_quality, comment)
        return data

    def qualityComment(
        self,
        c_x: float,
        c_y: float,
        x: float,
        y: float,
        a: str,
        signal: float,
        noise: float,
        ratio: float = 10,
    ) -> tuple[str, str]:
        if (abs(c_x - x) > 3 or abs(c_y - y) > 3) and a != "Alan Marscher":
            dr = np.sqrt((c_x - x) * (c_x - x) + (c_y - y) * (c_y - y))
            return (0, f"distance from map center to map max {dr} pixels")
        elif signal / noise <= ratio:
            return (0, f"snr = {signal / noise:.3f}")
        return (1, "")

    def getModels(self) -> pd.DataFrame:
        models = pd.DataFrame()
        keys = [FLUX, DELTAX, DELTAY, MAJOR_AX, MINOR_AX, POSANGLE, TYPE_OBJ]
        new_keys = [FLUX, DELTAX, DELTAY, "MAJOR_AX", "MINOR_AX", POSANGLE, "TYPE_OBJ"]

        # If multiple CC tables, using the first one
        field_num = self._cc_header[TFIELDS]
        for field, key, new_key in zip(range(field_num), keys, new_keys):
            models[new_key] = self._cc_data[key].tolist()

        if not (field_num == 7 or field_num == 3):
            raise FitsError("Wrong number of columns in CC table", self.file_name)
        return models
