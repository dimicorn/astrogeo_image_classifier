import logging
from astropy.io import fits
import numpy as np
from utils.fits import Fits
from utils.consts import (
    PRIMARY,
    AIPS_FQ,
    AIPS_AN,
    DATE_OBS,
    OBJECT,
    GCOUNT,
    NO_IF,
    UU,
    VV,
    IF_FREQ,
    CH_WIDTH,
)


logger = logging.getLogger(__name__)


class UVFits(Fits):
    def __init__(self, file_name: str) -> None:
        self.file_name_w_path = file_name
        self.file_name = file_name.split("/")[-1]
        self._uv_header, self._uv_data = None, None
        self._freq_header, self._freq_data = None, None
        self._antenna_header, self._antenna_data = None, None
        self._an_tables = None
        self._X = None

        self.ampl, self.phase = None, None

        with fits.open(file_name) as f:
            f.verify("fix")
            self.hdulist = f
            self._uv_header = f[PRIMARY].header
            self._uv_data = f[PRIMARY].data

            self.sanityCheck(f)

            if len(f) < 3:
                logger.error(f"Missing FQ or AN table in UV file: {self.file_name}")
            elif len(f) == 3:
                self._freq_header = f[AIPS_FQ].header
                self._freq_data = f[AIPS_FQ].data
                try:
                    self._antenna_header = f[AIPS_AN].header
                    self._antenna_data = f[AIPS_AN].data
                except KeyError:
                    self._antenna_header = f["AIPS NX"].header
                    self._antenna_data = f["AIPS NX"].data
                self._an_tables = 1
            else:
                # Assuming that other AN tables are the same,
                # no need to store them
                self._freq_header = f[AIPS_FQ].header
                self._freq_data = f[AIPS_FQ].data
                self._antenna_header = f[AIPS_AN].header
                self._antenna_data = f[AIPS_AN].data
                self._an_tables = len(f) - 2
                logger.warning(f"{self.file_name} has multiple AN tables")

        self.antennas = len(fits.getdata(file_name, extname=AIPS_AN))  # from Ilya
        self.freq = self.getFreq()
        self.date = self._uv_header[DATE_OBS]
        self.object = self._uv_header[OBJECT]
        self.uvData()

    def uvData(self) -> np.ndarray:
        """Reading UV data"""
        if self._X is None:
            data = self._uv_data
            gcount = self._uv_header[GCOUNT]
            if_nums = self._freq_header[NO_IF]
            if_freq = self._freq_data[IF_FREQ]

            uu, vv = [], []
            try:
                data[UU], data[VV]
                uu_key, vv_key = UU, VV
            except KeyError:
                try:
                    data["UU--"], data["VV--"]
                    uu_key, vv_key = "UU--", "VV--"
                except KeyError:
                    try:
                        data["UU---SIN"], data["VV---SIN"]
                        uu_key, vv_key = "UU---SIN", "VV---SIN"
                    except KeyError:
                        logger.warning(f"{self.file_name}: has weird UU and VV keys")

            if if_nums == 1:
                for ind in range(gcount):
                    for if_num in range(if_nums):
                        u = data[uu_key][ind] * (self.freq + if_freq[if_num])
                        v = data[vv_key][ind] * (self.freq + if_freq[if_num])
                        uu.append(u)
                        vv.append(v)
            elif if_nums > 1:
                for ind in range(gcount):
                    for if_num in range(if_nums):
                        u = data[uu_key][ind] * (self.freq + if_freq[0][if_num])
                        v = data[vv_key][ind] * (self.freq + if_freq[0][if_num])
                        uu.append(u)
                        vv.append(v)

            vis = data.data[:, 0, 0, :, 0, 0, 0] + data.data[:, 0, 0, :, 0, 0, 1] * 1j
            ampl = np.absolute(vis).flatten()
            phase = np.angle(vis).flatten()

            X = np.array([np.array(uu), np.array(vv), np.array(ampl), np.array(phase)])
            X_sym = np.copy(X)
            X_sym[0] = -1 * X_sym[0]
            X_sym[1] = -1 * X_sym[1]
            self._X = np.append(X.T, X_sym.T, axis=0).T

        return self._X

    def getSQLParams(self) -> tuple:
        """object_name, obs_date, freq,
        obs_author, file_name, min_uv_radius, max_uv_radius,
        visibilities, max_amplitude, min_amplitude, mean_amplitude,
        median_amplitude, freq_band, antennas, antenna_tables, uv_quality, comment"""
        header_data = self.headerData()
        radius = np.sqrt(self._X[0] * self._X[0] + self._X[1] * self._X[1])
        min_radius, max_radius = float(np.min(radius)), float(np.max(radius))
        ampl = self._X[2]
        ampl_data = (
            float(np.min(ampl)),
            float(np.max(ampl)),
            float(np.mean(ampl)),
            float(np.median(ampl)),
        )
        freq_ch_sum = float(np.sum(self.uvDataKeyCheck(CH_WIDTH)))  # freq band

        data = header_data + (min_radius, max_radius, self._X.shape[1]) + ampl_data
        data += (freq_ch_sum, self.antennas, self._an_tables)
        return data

    def getQualityParams(self) -> tuple[int]:
        visibilities = self._X.shape[1]
        antennas = self.antennas
        return visibilities, antennas
