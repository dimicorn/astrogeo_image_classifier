MAP_DIR = 'maps'
UV_DIR = 'uv_plots'

VIS_FITS = 'vis.fits'
MAP_FITS = 'map.fits'

CENTER = 'center'
LEFT = 'left'
RIGHT = 'right'
DOT = '.'

COLOR = 'blue'
CMAP = 'magma'

PRIMARY = 'PRIMARY'
AIPS_AN = 'AIPS AN'
AIPS_FQ = 'AIPS FQ'
AIPS_CC = 'AIPS CC'
EXTNAME = 'EXTNAME'

'''META DATA or just random shit'''
AUTHOR = 'AUTHOR'
GCOUNT = 'GCOUNT'
SIMPLE = 'SIMPLE'
NAXIS = 'NAXIS'

'''UV PRIMARY TABLE HEADER KEYS'''
# EXTNAME # ’AIPS UV’
OBJECT = 'OBJECT' # Source name
TELESCOP = 'TELESCOP' # Telescope name
INSTRUME = 'INSTRUME' # Instrument name (receiver or ?)
DATE_OBS = 'DATE-OBS' # Observation date
DATE_MAP = 'DATE-MAP' # File processing date # not found
BSCALE = 'BSCALE' # 1.0
BZERO = 'BZERO' # 0.0
BUNIT = 'BUNIT' # units, usually ’UNCALIB’ or ’JY’
EQUINOX = 'EQUINOX' # Equinox of source coordinates and uvw # not found
ALTRPIX = 'ALTRPIX' # Reference pixel for velocity

'''UV PRIMARY TABLE DATA KEYS'''
UU = 'UU' # u baseline coordinate [seconds]
VV = 'VV' # v baseline coordinate [seconds]
WW = 'WW' # w baseline coordinate [seconds]
DATE = 'DATE' # Julian date [days]
BASELINE = 'BASELINE' # Baseline number
# keys below weren't found
SOURCE = 'SOURCE' # Source ID number
FREQSEL = 'FREQSEL' # Frequency setup ID number
VISIBILITIES = 'VISIBILITIES' # Fringe visibility data [Jy]

# TODO: try to find all keys below

'''FQ TABLE HEADER KEYS'''
# EXTNAME # ’AIPS FQ’
NO_IF = 'NO_IF' # Number IFs (n_IF)

'''FQ TABLE DATA KEYS'''
FRQSEL = 'FRQSEL' # frequency setup ID number
IF_FREQ = 'IF FREQ' # Frequency offset [Hz]
CH_WIDTH = 'CH WIDTH' # Spectral channel separation [Hz]
TOTAL_BANDWIDTH = 'TOTAL BANDWIDTH' # Total width of spectral window [Hz]
SIDEBAND = 'SIDEBAND' # Sideband

'''AN TABLE HEADER KEYS'''
# EXTNAME # ’AIPS AN’
EXTVER = 'EXTVER' # Subarray number
ARRAYX = 'ARRAYX' # x coordinate of array center [meters]
ARRAYY = 'ARRAYY' # y coordinate of array center [meters]
ARRAYZ = 'ARRAYZ' # z coordinate of array center [meters]
GSTIAO = 'GSTIAO' # GST at 0h on reference date [degrees]
DEGPDY = 'DEGPDY' # Earth’s rotation rate [degrees/day]
FREQ = 'FREQ' # Reference frequency [Hz]
RDATE = 'RDATE' # Reference date
POLARX = 'POLARX' # x coordinate of North Pole [arc seconds]
POLARY = 'POLARY' # y coordinate of North Pole [arc seconds]
UT1UTC = 'UT1UTC' # UT1 - UTC [sec]
DATUTC = 'DATUTC' # time system - UTC [seconds]
TIMESYS = 'TIMESYS' # Time system
ARRNAM = 'ARRNAM' # Array name
XYZHAND = 'XYZHAND' # Handedness of station coordinates
FRAME = 'FRAME' # Coordinate frame
NUMORB = 'NUMORB' # Number orbital parameters in table (n_orb)
# NO_IF # Number IFs (n_IF)
NOPCAL = 'NOPCAL' # Number of polarization calibration values / IF (n_pcal)
POLTYPE = 'POLTYPE' # Type of polarization calibration
# ’APPROX’ - Linear approximation for circular feeds
# ’X-Y LIN’ - Linear approximation for linear feeds
# ’ORI-ELP’ - Orientation and ellipticity
# ’VLBI’ - VLBI solution form
FREQID = 'FREQID' # Frequency setup number

'''AN TABLE DATA KEYS'''
ANNAME = 'ANNAME' # Antenna name
STABXYZ = 'STABXYZ' # Antenna station coordinates (x, y, z) [meters]
ORBPARM = 'ORBPARM' # Orbital parameters
# Index 1 Semi-major axis of orbit (a) [meters]
# Index 2 Ellipticity of orbit (e)
# Index 3 Inclination of the orbit to the celestial equator (i) [degrees]
# Index 4 The right ascension of the ascending node (Omega) [degrees]
# Index 5 The argument of the perigee (omega) [degrees]
# Index 6 The mean anomaly (M) [degrees]
NOSTA = 'NOSTA' # Antenna number
MNTSTA = 'MNTSTA' # Mount type
STAXOF = 'STAXOF' # Axis offset [meters]
POLTYA = 'POLTYA' # ’R’, ’L’, feed A
POLAA = 'POLAA' # Position angle feed A [degrees]
POLCALA = 'POLCALA' # Calibration parameters feed A
POLTYB = 'POLTYB' # ’R’, ’L’, polarization 2
POLAB = 'POLAB' # Position angle feed B [degrees]
POLCALB = 'POLCALB' # Calibration parameters feed B

# TODO: add CC TABLE KEYS