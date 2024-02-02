import numpy as np
from astropy.io import fits
from consts import *
import pandas as pd


class FitsError(Exception):
    '''Base class for exceptions in this module.'''

    def __init__(self, message: str, file_name: str) -> None:
        self.file_name = file_name
        self.message = message

    def __str__(self) -> str:
        return f'{self.message}: {self.file_name}'

class Fits(object):
    hdulist = None
    file_name = None

    def sanity_check(self, f):
        if not f[PRIMARY].header[SIMPLE]:
            # raise FitsError('Non standard Fits file', self.file_name)
            print(f'Non standard Fits file {self.file_name}')
        
        header = f[PRIMARY].header
        obj_name_1 = header[OBJECT]
        file_name = self.file_name
        obj_name_2 = file_name.split('_')[0]
        if obj_name_1 != obj_name_2:
            # raise FitsError('Object name does not correspond to one in file name', self.file_name)
            print(f'Object name does not correspond to one in file name {self.file_name}')
        
        folder_name = self.file_name_w_path.split('/')[-2]
        if obj_name_1 != folder_name:
            # raise FitsError(f'Object {obj_name_1} in the {folder_name}/ directory', self.file_name)
            print(f'Object {obj_name_1} in the {folder_name}/ directory {self.file_name}')
        
        freq_bands = {'L': (1, 1.8), 'S': (1.8, 2.8), 'C': (2.8, 7), 'X': (7, 9), 'U': (9, 17), 'K': (17, 26),
                  'Q': (26, 50), 'W': (50, 100), 'G': (100, 250)}
        
        freq_band = file_name.split('_')[1]
        freq_lower, freq_upper = freq_bands[freq_band][0], freq_bands[freq_band][1]
        freq = self.get_freq() * 1e-9
        if not (freq_lower <= freq and freq <= freq_upper):
            # raise FitsError(f'Wrong FREQ band ({freq_band}) in file name, frequency value {freq} GHz', self.file_name)
            print(f'Wrong FREQ band ({freq_band}) in file name, frequency value {freq} GHz, {self.file_name}')

    def header_data(self) -> tuple:
        '''Reading PRIMARY table header'''
        header = self.hdulist[PRIMARY].header
        return (header[OBJECT], header[DATE_OBS], self.get_freq(), header[AUTHOR], self.file_name.split('/')[-1])

    def get_freq(self) -> float:
        # FIXME: Refactor this plz
        header = self.hdulist[PRIMARY].header
        for i in range(1, header[NAXIS] + 1):
            try:
                if header[f'CTYPE{i}'] == FREQ:
                    return header[f'CRVAL{i}']
            except KeyError:
                ...
        raise FitsError('No CTYPE_i == FREQ was found', self.file_name)
    
    def header_key_check(self, key):
        header = self._map_header
        try:
            return header[key]
        except KeyError:
            print(f'Caution: {self.file_name} has no {key} key')
            return -1
    
    def uv_data_key_check(self, key):
        data = self._freq_data
        try:
            return data[key]
        except KeyError:
            print(f'Caution: {self.file_name} has no {key} key')
            return -1

    def print_header(self) -> None:
        '''Printing header of the PRIMARY table'''
        header = self.hdulist[PRIMARY].header
        for key in header.keys():
            print(f'{key}\t{header[key]}')
    
    def info(self) -> None:
        self.hdulist.info()

class UVFits(Fits):
    def __init__(self, file_name: str) -> None:
        self.file_name_w_path = file_name
        self.file_name = file_name.split('/')[-1]
        self._uv_header, self._uv_data = None, None
        self._freq_header, self._freq_data = None, None
        self._antenna_header, self._antenna_data = None, None
        self._an_tables = None
        self._X = None

        self.ampl, self.phase = None, None

        with fits.open(file_name) as f:
            f.verify('fix')
            self.hdulist = f
            self._uv_header = f[PRIMARY].header
            self._uv_data = f[PRIMARY].data
            
            self.sanity_check(f)
            
            if len(f) < 3:
                raise FitsError('Missing FQ or AN table in UV file', self.file_name)
            elif len(f) == 3:
                self._freq_header = f[AIPS_FQ].header
                self._freq_data = f[AIPS_FQ].data
                try:
                    self._antenna_header = f[AIPS_AN].header
                    self._antenna_data = f[AIPS_AN].data
                except KeyError:
                    self._antenna_header = f['AIPS NX'].header
                    self._antenna_data = f['AIPS NX'].data
                self._an_tables = 1
            else:
                # Assuming that other AN tables are the same, no need to store them
                self._freq_header = f[AIPS_FQ].header
                self._freq_data = f[AIPS_FQ].data
                self._antenna_header = f[AIPS_AN].header
                self._antenna_data = f[AIPS_AN].data
                self._an_tables = len(f) - 2
                print(f'Caution: {self.file_name} has multiple AN tables')
        
        self.freq = self.get_freq()
        self.date = self._uv_header[DATE_OBS]
        self.object = self._uv_header[OBJECT]
        self.uv_data()

    def uv_data(self) -> np.array:
        '''Reading UV data'''
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
                    data['UU--'], data['VV--']
                    uu_key, vv_key = 'UU--', 'VV--'
                except KeyError:
                    try:
                        data['UU---SIN'], data['VV---SIN']
                        uu_key, vv_key = 'UU---SIN', 'VV---SIN'
                    except KeyError: print(f'Caution: {self.file_name} has weird UU and VV keys')

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

    def get_sql_params(self) -> tuple:
        '''object_name, obs_date, freq,
        obs_author, file_name, min_uv_radius, max_uv_radius,
        visibilities, max_amplitude, min_amplitude, mean_amplitude,
        median_amplitude, freq_band, antenna_tables, uv_quality, comment'''
        header_data = self.header_data()
        radius = np.sqrt(self._X[0] * self._X[0] + self._X[1] * self._X[1])
        min_radius, max_radius = np.min(radius), np.max(radius)
        ampl = self._X[2]
        ampl_data = (np.min(ampl), np.max(ampl), np.mean(ampl), np.median(ampl))
        freq_ch_sum = np.sum(self.uv_data_key_check(CH_WIDTH)) # freq band
        uv_quality = 'n/a'
        comment = uv_quality

        data = header_data + (min_radius, max_radius, self._X.shape[1]) + ampl_data
        data += (freq_ch_sum, self._an_tables, uv_quality, comment)
        return data

class MapFits(Fits):
    def __init__(self, file_name) -> None:
        self.file_name_w_path = file_name
        self.file_name = file_name.split('/')[-1]
        self._map_header, self._map_data = None, None
        self._cc_header, self._cc_data = None, None
        self._cc_tables = None

        with fits.open(file_name) as f:
            f.verify('fix')
            self.hdulist = f
            self._map_header = f[PRIMARY].header
            self._map_data = f[PRIMARY].data

            self.sanity_check(f)

            if len(f) == 1:
                print(f'Caution: {self.file_name} has no CC tables')
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
                print(f'Caution: {self.file_name} has multiple CC tables')
        
        self.date = self._map_header[DATE_OBS]
        self.object = self._map_header[OBJECT]
        self.freq = self.get_freq()

    def map_data(self):
        return self._map_data

    def get_parameters(self) -> pd.DataFrame:
        ''' get some parameters from a header: CRVAL, CRPIX, FREQ, SOURCE, DATE-OBS'''
        header = self._map_header
        keys = [CRPIX1, CRPIX2, CDELT1, CDELT2, NAXIS1, NAXIS2, BMAJ, BMIN, BPA,
                OBJECT, DATE_OBS, CRVAL3]
        params = {key: np.array([header[key]]) for key in keys}
        params[CDELT1] *= 3.6e6
        params[CDELT2] *= 3.6e6
        return pd.DataFrame(params)
    
    def map_noise(self, data, k=0.1) -> float:
        # TODO: check indexes
        # borders
        b1, b2 = int(k * data.shape[0]), int(k * data.shape[1])
        b3, b4 = int((1-k) * data.shape[0]), int((1-k) * data.shape[1])
        upper_left = np.std(data[:b1, :b2])
        upper_right = np.std(data[b3:, :b2])
        down_left = np.std(data[:b1, b4:])
        down_right = np.std(data[b3:, b4:])
        noise = np.median([upper_left, upper_right, down_left, down_right])
        return noise
    
    def get_sql_params(self) -> tuple:
        '''
        object_name, obs_date, freq, obs_author, file_name, 
        map_max, mapc_x, mapc_y, map_max_x, map_max_y, map_max_x_mas, map_max_y_mas, noise_level, 
        map_size_x, map_size_y, pixel_size_x, pixel_size_y, b_maj, b_min, b_pa, cc_tables, 
        map_quality, comment
        '''
        header_data = self.header_data()
        header, cc_tables = self._map_header, self._cc_tables
        
        map_data = self.map_data().squeeze()
        map_max = self.header_key_check('DATAMAX')
        # map_max = np.max(map_data)
        mapc_x, mapc_y = self.header_key_check(CRPIX1), self.header_key_check(CRPIX2)
        pixel_size_x = self.header_key_check(CDELT1) * 3.6e6
        pixel_size_y = self.header_key_check(CDELT2) * 3.6e6
        ind = np.argmax(map_data)

        map_max_y, map_max_x = np.unravel_index(ind, map_data.shape) # строчки и столбцы
        noise_level = self.map_noise(map_data)
        map_max_x_mas, map_max_y_mas = map_max_x * pixel_size_x, map_max_y * pixel_size_y
        noise = (map_max, mapc_x, mapc_y, map_max_x, map_max_y, map_max_x_mas, map_max_y_mas, noise_level)

        map_size_x, map_size_y = self.header_key_check(NAXIS1), self.header_key_check(NAXIS2)
        bmaj = self.header_key_check(BMAJ)
        bmin = self.header_key_check(BMIN)
        bpa = self.header_key_check(BPA)
        
        map_params = (map_size_x, map_size_y, pixel_size_x, pixel_size_y, 
                      bmaj, bmin, bpa, cc_tables)

        map_quality = 'n/a'
        comment = map_quality
        data = header_data + noise + map_params + (map_quality, comment)
        return data

    def get_models(self) -> pd.DataFrame:
        models = pd.DataFrame()
        keys = [FLUX, DELTAX, DELTAY, MAJOR_AX, MINOR_AX, POSANGLE, TYPE_OBJ]
        new_keys = [FLUX, DELTAX, DELTAY, 'MAJOR_AX', 'MINOR_AX', POSANGLE, 'TYPE_OBJ']
        
        # If multiple CC tables, using the first one
        field_num = self._cc_header[TFIELDS]
        for field, key, new_key in zip(range(field_num), keys, new_keys):
            models[new_key] = self._cc_data[key].tolist()

        if not (field_num == 7 or field_num == 3):
            raise FitsError('Wrong number of columns in CC table', self.file_name)
        return models
