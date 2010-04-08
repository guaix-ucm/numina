#
# Copyright 2008-2010 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 



from pyfits import Header, Card

from numina.image.storage import FITSCreator

_common_fits_headers = {
# FITS headers of the WCS HDU
'WCS': Header([
  Card('EXTNAME', 'WCSDVARR', ''),
  Card("EXTVER", 1, "Variance array version number"),
]),
# FITS headers of the Variance HDU
'VARIANCE': Header([
  Card('EXTNAME', 'ERROR DATA', ''),
  Card("EXTVER", 1, "Variance array version number"),
  Card("DATE", "", "Date FITS file was generated"),
  Card("DATE-OBS", "", "Date FITS file was generated"),
  Card("WCSDIM", 2, "Number of dimensions for WCS"),
  Card("CTYPE1", "LINEAR", "Type of WCS1"),
  Card("CTYPE2", "LINEAR", "Type of WCS2"),
  Card("BUNIT", "ADU", "Unit of the array of image data"),
  Card("CRPIX1", 1.0, "Pixel coordinate of reference point"),
  Card("CRPIX2", 1.0, "Pixel coordinate of reference point"),
]),
# FITS headers of the Primary HDU
'PRIMARY': Header([
#      Card("WCSDIM", 2, "Number of dimensions for WCS"),
#      Card("WCSAXES", 2, "Two coordinate axes"),
#      Card("WCSNAME  ", "Bidimensionnal Spatial representation", "Name of this system"),
#      Card("WCSNAME", "Spectral vs Spatial representation", "Name of this system"),
  Card("CRPIX1", 1.0, "Pixel coordinate of reference point"),
  Card("CRPIX2", 1.0, "Pixel coordinate of reference point"),
  Card("PC1_1", 1.0, "Linear transformation matrix element"),
  Card("PC1_2", 0.0, "Linear transformation matrix element"),
  Card("PC2_1", 0.0, "Linear transformation matrix element"),
  Card("PC2_2", 1.0, "Linear transformation matrix element"),
  Card("CUNIT2", "Pixel", "System unit"),
  Card("CTYPE1", "LINEAR", "Type of WCS1"),
  Card("CTYPE2", "LINEAR", "Type of WCS2"),
  Card("ORIGIN", "NUMINA", "FITS file originator"),
  Card("TELESCOP", "GTC", "Telescope id."),
  Card("OBSERVAT", "ORM", "Name of observatory"),
  Card("EQUINOX ", 2000.0, "[yr] Equinox of equatorial coordinates"),
  Card("LATITUDE", 28.762000, "[deg] Telescope latitude, +28:45:53.2"),
  Card("LONGITUD", 17.877639, "[deg] Telescope longitude, +17:52:39.5"),
  Card("HEIGHT", 2348, "[m] Height above sea level"),
  Card("SLATEL", "LP10.0", "Telescope name known to SLALIB"),
  Card("OBSGEOX", 5.753296428e+6, "[m] Observation X-position"),
  Card("OBSGEOY", 3.005451209e+6, "[m] Observation Y-position"),
  Card("OBSGEOZ", 9.694391412e+5, "[m] Observation Z-position"),
  Card("PROP_ID", " ", "Proposal identification."),
  Card("PROP_PI", "", "PI of proposal."),
  Card("OBSERVER", "NUMINA", "Name of Observer"),
  Card("RUN", 0, "Run number"),
  Card("OBSTYPE", "", "Type of observation, e.g. TARGET"),
  Card("IMAGETY", "", "Type of observation, e.g. object"),
  Card("RA", "00:00:00.000", "RA of the instrument scientific aperture"),
  Card("DEC", "00:00:00.000", "DEC of the instrument scientific aperture"),
  Card("RADECSYS", "FK5", "System of coordinates of scientific aperture"),
  Card("AIRMASS1", 1., "Effective mean airmass at start"),
  Card("AIRMASS2", 1., "Effective mean airmass at end"),
  Card("FILTER", "", "Filter id."),
  Card("GRATING", "", "Grating id."),
  Card("CAT-NAME", "", "Target input-catalog name"),
  Card("OBJECT", "", "Name of observed object"),
  Card("CAT-RA", "UNDEF", "Target Right-Ascension"),
  Card("CAT-DEC", "UNDEF", "Target Declination"),
  Card("CAT-EQUI", "J2000.0", "Equinox of target coordinates"),
  Card("CAT-EPOC", 2000.0, "Target epoch of proper motions"),
  Card("PM-RA", 0.0, "[s/yr] Target proper-motion RA"),
  Card("PM-DEC", 0.0, "[arcsec/yr] Target proper-motion"),
  Card("PARALLAX", 0.0, "[arcsec] Target Parallax"),
  Card("RADVEL", 0.0, "[km/s] Target radial velocity"),
  # Instrument
  Card("INSTRUME", "EMIR", "Name of the Instrument"),
  Card("FSTATION", "NASMYTH", "Focal station of observation"),
  Card("PLATESCA", 1.502645, "[d/m] Platescale Card(5.41 arcsec/mm)"),
  Card("ELAPSED", 0, "[s] Time from end of CLR to shutter close"),
  Card("DARKTIME", 0, "[s] Time from end of CLR to start of r/o"),
  Card("EXPOSED", 0, "[s] Time between correlated reads"),
  Card("EXPTIME", 0, "[s] Time between correlated reads"),
  Card("BUNIT", "ADU", "Unit of the array of image data"),
  Card("GAIN", 3.6, "Electrons per ADU"),
  Card("READNOIS", 2.16, "Readout noise in electrons per pix"),
  # Readout mode
  Card("READMODE", "FOWLER", "Detector read-out mode"),
  Card("READSCHM", "PERLINE", "Read-out scheme of the detector"),
  Card("READNUM", 0, "Number of reads in the mode"),
  Card("READREPT", 1, "Number of times to repeat the read-out"),
  Card("DATE", "", "Date FITS file was generated"),
  Card("DATE-OBS", "", "Date FITS file was generated")
  ])
}

_image_extra_fits_headers = {
# FITS headers of the Variance HDU
'VARIANCE': Header([
  Card("CRVAL1", 1, "Starting pixel in X spatial direction"),
  Card("CRVAL2", 1, "Starting pixel in Y spatial direction"),
  Card("CDELT1", 1, "Spatial interval in pixels"),
  Card("CDELT1", 2, "Spatial interval in pixels"),
]),
# FITS headers of the Primary HDU
'PRIMARY': Header([
  Card("CUNIT1", "Pixel", "System unit"),
  Card("CRVAL1  ", 1, "Starting pixel in X spatial sirection"),
  Card("CRVAL2  ", 1, "Starting pixel in Y spatial direction"),
  Card("CDELT1  ", 1, "Spatial interval in pixels"),
  Card("CDELT2  ", 1, "Spatial interval in pixels"),
  Card("OBS_MODE", "Direct Imaging", "Observing mode id."),
  ])
}

_spectrum_extra_fits_headers = {
# FITS headers of the Variance HDU
'VARIANCE': Header([
  Card("CUNIT1", "Angstrom", "System unit"),
  Card("CRVAL1", "", "Starting wavelength of output spectra"),
  Card("CRVAL2", 1, "Starting pixel in spatial direction"),
  Card("CDELT1", 2, "Wavelength interval between pixels"),
  Card("CDELT2", 1, "Spatial interval in pixels"),
]),
# FITS headers of the Primary HDU
'PRIMARY': Header([
  Card("CUNIT1", "Angstrom", "System unit"),
  Card("CRVAL1", "", "Starting wavelength of output spectra"),
  Card("CRVAL2", 1, "Starting pixel in spatial direction"),
  Card("CDELT1", 2, "Wavelength interval between pixels"),
  Card("CDELT2", 1, "Spatial interval in pixels"),
  Card("OBS_MODE", "Long-slit Spectroscopy", "Observing mode id."),
])
}


def _merge(headers1, headers2):
    '''Merge the contents of header2 with header1.'''
    result = {}
    
    for key in headers1:
        if key in headers2:
            fhead = headers1[key].copy()
            fhead.ascard.extend(headers2[key].ascard)
            result[key] = fhead

    return result

_image_fits_headers = _merge(_common_fits_headers, _image_extra_fits_headers)
_spectrum_fits_headers = _merge(_common_fits_headers, _spectrum_extra_fits_headers)

class EmirImageCreator(FITSCreator):
    '''Builder of Emir direct image.'''
    def __init__(self): 
        super(EmirImageCreator, self).__init__(_image_fits_headers)
        
class EmirSpectrumCreator(FITSCreator):
    '''Builder of Emir spectrum.'''
    def __init__(self): 
        super(EmirSpectrumCreator, self).__init__(_spectrum_fits_headers)
