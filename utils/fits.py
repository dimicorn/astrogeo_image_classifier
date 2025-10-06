import logging
from utils.consts import PRIMARY, SIMPLE, OBJECT, DATE_OBS, AUTHOR, NAXIS, FREQ


logger = logging.Logger(__name__)


class Fits(object):
    hdulist = None
    file_name = None

    def sanityCheck(self, f) -> None:
        if not f[PRIMARY].header[SIMPLE]:
            logger.error(f"Non standard Fits file {self.file_name}")

        header = f[PRIMARY].header
        obj_name_1 = header[OBJECT]
        file_name = self.file_name
        obj_name_2 = file_name.split("_")[0]
        if obj_name_1 != obj_name_2:
            logger.error(
                f"Object name does not correspond to one in file name {self.file_name}"
            )

        folder_name = self.file_name_w_path.split("/")[-2]
        if obj_name_1 != folder_name:
            logger.error(
                f"Object {obj_name_1} in the {folder_name}/ directory {self.file_name}"
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
        freq = self.getFreq() * 1e-9
        if not (freq_lower <= freq and freq <= freq_upper):
            logger.error(
                f"Wrong FREQ band ({freq_band}) in file name, "
                f"frequency value {freq} GHz, {self.file_name}"
            )

    def headerData(self) -> tuple:
        """Reading PRIMARY table header"""
        header = self.hdulist[PRIMARY].header
        return (
            header[OBJECT],
            header[DATE_OBS],
            self.getFreq(),
            header[AUTHOR],
            self.file_name.split("/")[-1],
        )

    def getFreq(self) -> float:
        # FIXME: Refactor this plz
        header = self.hdulist[PRIMARY].header
        for i in range(1, header[NAXIS] + 1):
            try:
                if header[f"CTYPE{i}"] == FREQ:
                    return header[f"CRVAL{i}"]
            except KeyError:
                ...
        raise logger.error("No CTYPE_i == FREQ was found", self.file_name)

    def headerKeyCheck(self, key) -> float:
        header = self._map_header
        try:
            return header[key]
        except KeyError:
            logger.warning(f"{self.file_name} has no {key} key")
            return -1

    def uvDataKeyCheck(self, key) -> float:
        data = self._freq_data
        try:
            return data[key]
        except KeyError:
            logger.warning(f"{self.file_name} has no {key} key")
            return -1

    def printHeader(self) -> None:
        """Printing header of the PRIMARY table"""
        header = self.hdulist[PRIMARY].header
        for key in header.keys():
            logger.info(f"{key}\t{header[key]}")

    def info(self) -> None:
        self.hdulist.info()
