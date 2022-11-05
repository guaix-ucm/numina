#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Corrections of wavelength calibration"""

import logging
import warnings

import astropy.wcs
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.time
import astropy.constants as cons

_logger = logging.getLogger(__name__)


def header_add_barycentric_correction(hdr, key='B', out=None):
    """Add WCS keywords with barycentric correction

    Raises
    ------
    KeyError
        If a required keyword is missing
    TypeError
        If the header does not contain a spectral axis
    """

    # Header must have DATE-OBS
    if 'DATE-OBS' not in hdr:
        raise KeyError("Keyword 'DATE-OBS' not found.")
    # Header must contain a primary WCS
    # Header must contain RADEG and DECDEG

    if 'OBSGEO-X' not in hdr:
        warnings.warn('OBSGEO- keywords not defined, using default values for GTC', RuntimeWarning)
        # Geocentric coordinates of GTC
        hdr['OBSGEO-X'] = 5327285.0921
        hdr['OBSGEO-Y'] = -1718777.1125
        hdr['OBSGEO-Z'] = 3051786.7327

    # Get main WCS
    wcs0 = astropy.wcs.WCS(hdr)
    if wcs0.wcs.spec == -1:
        # We don't have a spec axis
        raise TypeError('Header does not contain spectral axis')
    gtc = EarthLocation.from_geocentric(wcs0.wcs.obsgeo[0], wcs0.wcs.obsgeo[1], wcs0.wcs.obsgeo[2], unit='m')
    date_obs = astropy.time.Time(wcs0.wcs.dateobs, format='fits')
    # if frame='fk5', we need to pass the epoch and equinox
    sc = SkyCoord(ra=hdr['RADEG'], dec=hdr['DECDEG'], unit='deg')
    rv = sc.radial_velocity_correction(obstime=date_obs, location=gtc)
    factor = (1 + rv / cons.c).to('').value

    if out is None:
        out = hdr

    out[f'WCSNAME{key}'] = 'Barycentric correction'
    # out['CNAME1{}'.format(key)] = 'AxisV'
    out[f'CTYPE1{key}'] = hdr['CTYPE1']
    out[f'CRPIX1{key}'] = hdr['CRPIX1']
    out[f'CRVAL1{key}'] = hdr['CRVAL1'] * factor
    out[f'CDELT1{key}'] = hdr['CDELT1'] * factor
    out[f'CUNIT1{key}'] = hdr['CUNIT1']

    for keyword in ['CRPIX2', 'CRVAL2', 'CDELT2']:
        out[f'{keyword}{key}'] = hdr[f'{keyword}']

    out[f'VELOSYS{key}'] = rv.to('m / s').value
    out[f'SPECSYS{key}'] = 'BARYCENT'
    out[f'SSYSOBS{key}'] = 'TOPOCENT'
    return out
