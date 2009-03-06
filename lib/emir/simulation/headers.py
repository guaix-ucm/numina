#
# Copyright 2008-2009 Sergio Pascual
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

# $Id$

import pyfits

__version__ = "$Revision$"

default_fits_headers = {
# FITS headers of the WCS HDU
'wcs': pyfits.Header([
  pyfits.Card('EXTNAME','WCSDVARR', ''),
  pyfits.Card("EXTVER", 1, "Variance array version number"),
]),
# FITS headers of the Variance HDU
'variance': pyfits.Header([
  pyfits.Card('EXTNAME','ERROR DATA',''),
  pyfits.Card("EXTVER", 1, "Variance array version number"),
  pyfits.Card("DATE", "", "Date FITS file was generated"),
  pyfits.Card("DATE-OBS", "", "Date FITS file was generated"),
  pyfits.Card("WCSDIM", 2, "Number of dimensions for WCS"),
  pyfits.Card("CTYPE1", "LINEAR", "Type of WCS1"),
  pyfits.Card("CTYPE2", "LINEAR", "Type of WCS2"),
  pyfits.Card("BUNIT","ADU","Unit of the array of image data"),
  pyfits.Card("CRPIX1", 1.0, "Pixel coordinate of reference point"),
  pyfits.Card("CRPIX2", 1.0, "Pixel coordinate of reference point"),
  pyfits.Card("CRVAL1", 1, "Starting pixel in X spatial direction"), 
  pyfits.Card("CRVAL2", 1, "Starting pixel in Y spatial direction"),
  pyfits.Card("CDELT1", 1, "Spatial interval in pixels"),
  pyfits.Card("CDELT1", 2, "Spatial interval in pixels"),
  #      pyfits.Card("CRVAL1", "", "Starting wavelength of output spectra"),
  #      pyfits.Card("CRVAL2", 1, "Starting pixel in spatial direction"),
  #      pyfits.Card("CDELT1", 2, "Wavelength interval between pixels")
  #      pyfits.Card("CDELT2", 1, "Spatial interval in pixels"),
]),
# FITS headers of the Primary HDU
'primary': pyfits.Header([
#      pyfits.Card("WCSDIM", 2, "Number of dimensions for WCS"),
#      pyfits.Card("WCSAXES", 2, "Two coordinate axes"),
#      pyfits.Card("WCSNAME  ", "Bidimensionnal Spatial representation", "Name of this system"),
#      pyfits.Card("WCSNAME", "Spectral vs Spatial representation", "Name of this system"),
  pyfits.Card("CRPIX1", 1.0, "Pixel coordinate of reference point"),
  pyfits.Card("CRPIX2", 1.0, "Pixel coordinate of reference point"),
  pyfits.Card("PC1_1", 1.0, "Linear transformation matrix element"),
  pyfits.Card("PC1_2", 0.0, "Linear transformation matrix element"),
  pyfits.Card("PC2_1", 0.0, "Linear transformation matrix element"),
  pyfits.Card("PC2_2", 1.0, "Linear transformation matrix element"),
  pyfits.Card("CUNIT1", "Pixel", "System unit"),
#      pyfits.Card("CUNIT1", "Angstrom", "System unit"),
  pyfits.Card("CUNIT2", "Pixel", "System unit"),
  pyfits.Card("CTYPE1", "LINEAR", "Type of WCS1"),
  pyfits.Card("CTYPE2", "LINEAR", "Type of WCS2"),
  pyfits.Card("CRVAL1  ", 1, "Starting pixel in X spatial sirection"),
  pyfits.Card("CRVAL2  ", 1, "Starting pixel in Y spatial direction"),
#      pyfits.Card("CRVAL1", "", "Starting wavelength of output spectra"),
#      pyfits.Card("CRVAL2", 1, "Starting pixel in spatial direction"),
  pyfits.Card("CDELT1  ", 1, "Spatial interval in pixels"),
  pyfits.Card("CDELT2  ", 1, "Spatial interval in pixels"),
#      pyfits.Card("CDELT1", 2, "Wavelength interval between pixels"),
#      pyfits.Card("CDELT2", 1, "Spatial interval in pixels"),
  pyfits.Card("ORIGIN", "RIME-Image Simulator", "FITS file originator"),
  pyfits.Card("TELESCOP", "GTC","Telescope id."),
  pyfits.Card("OBSERVAT", "ORM","Name of observatory"),
  pyfits.Card("EQUINOX ", 2000.0, "[yr] Equinox of equatorial coordinates"),
  pyfits.Card("LATITUDE", 28.762000, "[deg] Telescope latitude, +28:45:53.2"),
  pyfits.Card("LONGITUD", 17.877639, "[deg] Telescope longitude, +17:52:39.5"),
  pyfits.Card("HEIGHT", 2348, "[m] Height above sea level"),
  pyfits.Card("SLATEL", "LP10.0", "Telescope name known to SLALIB"),
  pyfits.Card("OBSGEOX", 5.753296428e+6, "[m] Observation X-position"),
  pyfits.Card("OBSGEOY", 3.005451209e+6, "[m] Observation Y-position"),
  pyfits.Card("OBSGEOZ", 9.694391412e+5, "[m] Observation Z-position"),
  pyfits.Card("PROP_ID", " ", "Proposal identification."),
  pyfits.Card("PROP_PI", "", "PI of proposal."),
  pyfits.Card("OBSERVER", "RIME", "Name of Observer"),
  pyfits.Card("RUN", 0, "Run number"),
#      pyfits.Card("OBS_MODE", "Long-slit Spectroscopy", "Observing mode id."),
  pyfits.Card("OBS_MODE", "Direct Imaging", "Observing mode id."),
  pyfits.Card("OBSTYPE", "","Type of observation, e.g. TARGET"),
  pyfits.Card("IMAGETY", "","Type of observation, e.g. object"),
  pyfits.Card("RA", "00:00:00.000", "RA of the instrument scientific aperture"),
  pyfits.Card("DEC", "00:00:00.000", "DEC of the instrument scientific aperture"),
  pyfits.Card("RADECSYS", "FK5", "System of coordinates of scientific aperture"),
  pyfits.Card("AIRMASS1", 1., "Effective mean airmass at start"),
  pyfits.Card("AIRMASS2", 1., "Effective mean airmass at end"),
  pyfits.Card("FILTER", "", "Filter id."),
  pyfits.Card("GRATING", "", "Grating id."),
  pyfits.Card("CAT-NAME", "", "Target input-catalog name"),
  pyfits.Card("OBJECT", "", "Name of observed object"),
  pyfits.Card("CAT-RA", "UNDEF", "Target Right-Ascension"),
  pyfits.Card("CAT-DEC","UNDEF", "Target Declination"),
  pyfits.Card("CAT-EQUI","J2000.0", "Equinox of target coordinates"),
  pyfits.Card("CAT-EPOC",2000.0, "Target epoch of proper motions"),
  pyfits.Card("PM-RA",0.0, "[s/yr] Target proper-motion RA"),
  pyfits.Card("PM-DEC",0.0, "[arcsec/yr] Target proper-motion"),
  pyfits.Card("PARALLAX",0.0, "[arcsec] Target Parallax"),
  pyfits.Card("RADVEL",0.0, "[km/s] Target radial velocity"),
  # Instrument
  pyfits.Card("INSTRUME", "EMIR","Name of the Instrument"),
  pyfits.Card("FSTATION", "NASMYTH","Focal station of observation"),
  pyfits.Card("PLATESCA", 1.502645,"[d/m] Platescale pyfits.Card(5.41 arcsec/mm)"),
  pyfits.Card("ELAPSED", 0,"[s] Time from end of CLR to shutter close"),
  pyfits.Card("DARKTIME", 0,"[s] Time from end of CLR to start of r/o"),
  pyfits.Card("EXPOSED", 0,"[s] Time between correlated reads"),
  pyfits.Card("EXPTIME", 0,"[s] Time between correlated reads"),
  pyfits.Card("BUNIT", "ADU","Unit of the array of image data"),
  pyfits.Card("GAIN", 3.6,"Electrons per ADU"),
  pyfits.Card("READNOIS", 2.16,"Readout noise in electrons per pix"),
  # Readout mode
  pyfits.Card("READMODE", "FOWLER","Detector read-out mode"),
  pyfits.Card("READSCHM", "PERLINE","Read-out scheme of the detector"),
  pyfits.Card("READNUM", 0,"Number of reads in the mode"),
  pyfits.Card("READREPT", 1,"Number of times to repeat the read-out"),
  pyfits.Card("DATE", "", "Date FITS file was generated"),
  pyfits.Card("DATE-OBS", "", "Date FITS file was generated")
  ])
}
    
