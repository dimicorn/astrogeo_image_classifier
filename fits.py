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

    def uv_header(self) -> tuple[str, str, str]:
        '''Reading PRIMARY table header'''
        header = self.hdulist[PRIMARY].header
        return (header[OBJECT], header[DATE_OBS], header[AUTHOR])

    def get_freq(self) -> float:
        header = self.hdulist[PRIMARY].header
        for i in range(1, header[NAXIS] + 1):
            if header[f'CTYPE{i}'] == FREQ:
                return header[f'CRVAL{i}']
        raise FitsError('No CTYPE_i == FREQ was found', self.file_name)
    
    def print_header(self) -> None:
        '''Printing header of the PRIMARY table'''
        header = self.hdulist[PRIMARY].header
        for key in header.keys():
            print(f'{key}\t{header[key]}')
    
    def info(self) -> None:
        self.hdulist.info()

class UVFits(Fits):
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self._uv_header, self._uv_data = None, None
        self._freq_header, self._freq_data = None, None
        self._antenna_header, self._antenna_data = None, None

        with fits.open(file_name) as f:
            f.verify('fix')
            self.hdulist = f
            self._uv_header = f[PRIMARY].header
            self._uv_data = f[PRIMARY].data

            if not f[PRIMARY].header[SIMPLE]:
                raise FitsError('Non standard Fits file', self.file_name)
            if len(f) < 3:
                raise FitsError('Missing FQ or AN table in UV file', self.file_name)
            elif len(f) == 3:
                self._freq_header = f[AIPS_FQ].header
                self._freq_data = f[AIPS_FQ].data
                self._antenna_header = f[AIPS_AN].header
                self._antenna_data = f[AIPS_AN].data
            else:
                # FIXME : Think of a method to store multiple AN tables
                self._freq_header = f[AIPS_FQ].header
                self._freq_data = f[AIPS_FQ].data
                self._antenna_header = f[AIPS_AN].header
                self._antenna_data = f[AIPS_AN].data
                print(f'Caution: {self.file_name} has multiple AN tables')
        
        self.freq = self._antenna_header[FREQ]
        self.date = self._uv_header[DATE_OBS]
        self.object = self._uv_header[OBJECT]

    def uv_data(self) -> tuple[np.array, np.array]:
        '''Reading UV data'''
        data = self._uv_data
        gcount = self._uv_header[GCOUNT]
        if_nums = self._freq_header[NO_IF]
        if_freq = self._freq_data[IF_FREQ]

        uu, vv = [], []
        for ind in range(gcount):
            for if_num in range(if_nums):
                uu.append(data[UU][ind] * (self.freq + if_freq[if_num]))
                vv.append(data[VV][ind] * (self.freq + if_freq[if_num]))
        
        return (np.array(uu), np.array(vv))

    def _read_fq_table(self) -> None:
        '''Reading FQ (frequency) table'''
        self.no_if = self.hdulist[AIPS_FQ].header[NO_IF]

        # Assuming we have only one frequency setup
        freq_data = self.hdulist[AIPS_FQ].data[0]

        if self.no_if == 1:
            self.freq_table.append(
                {
                    IF_FREQ: freq_data[IF_FREQ],
                    CH_WIDTH: freq_data[CH_WIDTH],
                    SIDEBAND: freq_data[SIDEBAND],
                }
            )
        elif self.no_if >= 2:
            for if_num in range(self.no_if):
                self.freq_table.append(
                    {
                        IF_FREQ: freq_data[IF_FREQ][if_num],
                        CH_WIDTH: freq_data[CH_WIDTH][if_num],
                        SIDEBAND: freq_data[SIDEBAND][if_num],
                    }
                )
        else:
            raise FitsError(
                f'Invalid NO_IF value: {self.no_if}', self.file_name
            )

    def _read_an_table(self) -> None:
        '''Reading AN (antenna) table'''
        an_data = []
        for sub in range(self.antenna_header[EXTVER]):
            for row in self.antenna_data:
                an_data.append([sub+1, row[NOSTA], row[ANNAME], ])
        self.an_df = pd.DataFrame(columns=[EXTVER, NOSTA, ANNAME], data=an_data)
        # print(self.an_df)

class MapFits(Fits):
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self._map_header, self._map_data = None, None
        self._cc_header, self._cc_data = None, None

        with fits.open(file_name) as f:
            f.verify('fix')
            self.hdulist = f
            self._map_header = f[PRIMARY].header
            self._map_data = f[PRIMARY].data
            if not f[PRIMARY].header[SIMPLE]:
                raise FitsError('Non standard Fits file', self.file_name)
            if len(f) == 1:
                print(f'Caution: {self.file_name} has no CC tables')
            elif len(f) == 2:
                self._cc_header = f[AIPS_CC].header
                self._cc_data = f[AIPS_CC].data
            else:
                raise FitsError('Too many CC table in file', self.file_name)
        
        self.date = self._map_header[DATE_OBS]
        self.object = self._map_header[OBJECT]
        self.freq = self.get_freq()
        # print(list(self._map_header.keys()))
        # print(self._map_header['PCOUNT'])

    def map_data(self):
        return self._map_data

    def get_parameters(self) -> pd.DataFrame:
        # map parameters
        ''' get some parameters from a header: CRVAL, CRPIX, FREQ, SOURCE, DATE-OBS'''
        header = self._map_header
        
        params = pd.DataFrame(columns = ['racenpix', 'deccenpix', # central pixel coords in px
                                        'rapixsize', 'decpixsize', # pixel size in degrees
                                        'ramapsize', 'decmapsize', # map size in px
                                        'bmaj', 'bmin', 'bpa', # degrees, to be checked
                                        'source', 'dateobs',
                                        'frequency', # Hz
                                        'masperpix', 'masperpixx', 'masperpixy'], # pixel size in mas
                            data = [[header['CRPIX1'], header['CRPIX2'],
                                header['CDELT1'], header['CDELT2'], 
                                header['NAXIS1'], header['NAXIS2'],
                                header['BMAJ'], header['BMIN'], header['BPA'],
                                header['OBJECT'], header['DATE-OBS'],
                                header['CRVAL3'], 
                                np.abs(header['CDELT1']) * 3.6e6, 
                                header['CDELT1'] * 3.6e6, header['CDELT2'] * 3.6e6]])
        # MASperPIX = np.abs(self.rm[0].header['CDELT1']*3.6e6)
        return params

    def models(self) -> None:
        header = self._cc_header
        print(list(header.keys()))
        # print(map[1].header['EXTVER'])
        # print(map[1].data['FLUX'])
        # print(map[1].data.field(0))
        # print(map[1].data['DELTAX'])
        # print(map[1].data['DELTAY'])
        # print(map[1].data['MAJOR AX'])
        # print(map[1].data['MINOR AX'])
        # print(map[1].data['POSANGLE'])
        # print(map[1].data['TYPE OBJ'])

        datum_type = 'TTYPE'
        for i in range(7):
            print(map[1].header[f'TTYPE{i+1}'], map[1].header[f'TUNIT{i+1}'])
    