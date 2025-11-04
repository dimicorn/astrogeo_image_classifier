import logging
from astropy.io.fits.hdu.hdulist import HDUList
from utils.consts import PRIMARY, SIMPLE, OBJECT, DATE_OBS, AUTHOR, NAXIS, FREQ


logger = logging.getLogger(__name__)


class Fits:
    def __init__(self) -> None:
        self.hdulist: HDUList = None
        self.file_name: str = None
        self.file_name_w_path: str = None

    def sanity_check(self, f) -> None:
        if not f[PRIMARY].header[SIMPLE]:
            logger.error("Non standard Fits file %s", self.file_name)

        header = f[PRIMARY].header
        obj_name_1: str = header[OBJECT]
        file_name = self.file_name
        obj_name_2 = file_name.split("_")[0]
        if obj_name_1 != obj_name_2:
            logger.error(
                "Object name does not correspond to one in file name %s",
                self.file_name,
            )

        folder_name = self.file_name_w_path.split("/")[-2]
        if obj_name_1 != folder_name:
            logger.error(
                "Object %s in the %s/ directory %s",
                obj_name_1,
                folder_name,
                self.file_name,
            )

        freq_bands = {
            "L": (1, 1.8),
            "S": (1.8, 2.8),
            "C": (2.8, 7),
            "X": (7, 9),
            "U": (9, 17),
            "K": (17, 26),
            "Q": (26, 50),
            "W": (50, 100),
            "G": (100, 250),
        }

        freq_band = file_name.split("_")[1]
        freq_lower, freq_upper = freq_bands[freq_band][0], freq_bands[freq_band][1]
        freq = self.get_freq() * 1e-9
        if not (freq_lower <= freq and freq <= freq_upper):
            logger.error(
                "Wrong FREQ band (%s) in file name, " "frequency value %.3f GHz, %s",
                freq_band,
                freq,
                self.file_name,
            )

    def header_data(self) -> tuple:
        """Reading PRIMARY table header"""
        header = self.hdulist[PRIMARY].header  # pylint: disable=unsubscriptable-object
        return (
            header[OBJECT],
            header[DATE_OBS],
            self.get_freq(),
            header[AUTHOR],
            self.file_name.split("/")[-1],
        )

    def get_freq(self) -> float:
        # FIXME: Refactor this plz
        header = self.hdulist[PRIMARY].header  # pylint: disable=unsubscriptable-object
        for i in range(1, header[NAXIS] + 1):
            try:
                if header[f"CTYPE{i}"] == FREQ:
                    return header[f"CRVAL{i}"]
            except KeyError:
                ...
        logger.error("No CTYPE_i == FREQ was found %s", self.file_name)

    def header_key_check(self, key) -> float:
        header = self._map_header
        try:
            return header[key]
        except KeyError:
            logger.warning("%s has no %s key", self.file_name, key)
            return -1

    def uv_data_key_check(self, key) -> float:
        data = self._freq_data
        try:
            return data[key]
        except KeyError:
            logger.warning("%s has no %s key", self.file_name, key)
            return -1

    def print_header(self) -> None:
        """Printing header of the PRIMARY table"""
        header = self.hdulist[PRIMARY].header  # pylint: disable=unsubscriptable-object
        for key in header.keys():
            logger.info("%s\t%s", key, header[key])

    def info(self) -> None:
        self.hdulist.info()
