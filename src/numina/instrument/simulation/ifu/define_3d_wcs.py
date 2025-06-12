#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Define a 3D WCS"""
# ToDo: this module should be moved to numina

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import numpy as np


def define_3d_wcs(naxis1_ifu, naxis2_ifu, skycoord_center, spatial_scale, wv_lincal, instrument_pa, verbose):
    """Define a 3D WCS.

    Parameters
    ----------
    naxis1_ifu : `~astropy.units.Quantity`
        NAXIS1 value of the WCS object (along slice).
    naxis2_ifu : `~astropy.units.Quantity`
        NAXIS2 value of the WCS object (perpendicular to the slice).
    skycoord_center : `~astropy.coordinates.sky_coordinate.SkyCoord`
        Coordinates at the center of the detector.
    spatial_scale : `~astropy.units.Quantity`
        Spatial scale per pixel.
    wv_lincal : `~.linear_wavelength_calibration.LinearWaveCal`
        Linear wavelength calibration object.
    instrument_pa : `~astropy.units.Quantity`
        Instrument Position Angle.
    verbose : bool
        If True, display additional information.

    Returns
    -------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.

    """

    # initial checks
    if not naxis1_ifu.unit.is_equivalent(u.pix):
        raise ValueError(f'Unexpected naxis1 unit: {naxis1_ifu.unit}')
    if not naxis2_ifu.unit.is_equivalent(u.pix):
        raise ValueError(f'Unexpected naxis2 unit: {naxis2_ifu.unit}')
    if not isinstance(skycoord_center, SkyCoord):
        raise ValueError(f'Expected SkyCoord instance not found: {skycoord_center} of type {type(skycoord_center)}')
    if not spatial_scale.unit.is_equivalent(u.deg / u.pix):
        raise ValueError(f'Unexpected spatial_scale unit: {spatial_scale.unit}')

    default_wv_unit = wv_lincal.default_wavelength_unit

    # Define FITS header
    # Note: when using CDi_j keywords, there is no need to define CDELTi
    #       keywords, since they are ignored by astropy.wcs.WCS.
    #
    #       The FITS standard defined 
    #       in https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf 
    #       (page 30) states that there are three possible conventions:
    #       1. The CDi_j keywords: represent a full linear transformation matrix, 
    #          including rotation, scaling, and possible distortion. Used when you 
    #          want to describe the complete transformation in a single matrix.
    #       2. The PCi_j + CDELTi keywords: PCi_j should represent a rotation 
    #          and distortion matrix without units, meaning it should be normalized.
    #          The CDELTi keywords should represent the pixel scale in the
    #          corresponding axis, with units.
    #       3. CDELTi + ROTA2 keywords (not recommended anymore).
    #       A potential problem with the returned wcs3d object is that
    #       when using wcs3d.to_header() the CDi_j keywords are replaced by
    #       PCi_j and CDELTi keywords. This is actually a problem because
    #       the PCi_j keywords are not defined in the FITS standard (the
    #       resulting matrix should be normalized, which is not; see
    #       https://github.com/astropy/astropy/issues/1084 for more details).
    #       A possible solution is to define a custom wcs_to_header_using_cd_keywords(wcs3d)
    #       function that returns the header with CDi_j keywords.
    #       This function should be used instead of wcs3d.to_header().
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = int(naxis1_ifu.value)
    header['NAXIS2'] = int(naxis2_ifu.value)
    header['NAXIS3'] = int(wv_lincal.naxis1_wavecal.value)
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CTYPE3'] = 'WAVE'
    header['CRVAL1'] = skycoord_center.ra.deg
    header['CRVAL2'] = skycoord_center.dec.deg
    header['CRVAL3'] = wv_lincal.crval1_wavecal.to(default_wv_unit).value
    header['CRPIX1'] = (naxis1_ifu.value + 1) / 2
    header['CRPIX2'] = (naxis2_ifu.value + 1) / 2
    header['CRPIX3'] = wv_lincal.crpix1_wavecal.value
    spatial_scale_deg_pix = spatial_scale.to(u.deg / u.pix).value
    header['CD1_1'] = (-spatial_scale_deg_pix) * np.cos(instrument_pa).value
    header['CD1_2'] = spatial_scale_deg_pix * np.sin(instrument_pa).value
    header['CD2_1'] = (-spatial_scale_deg_pix) * (-np.sin(instrument_pa).value)
    header['CD2_2'] = spatial_scale_deg_pix * np.cos(instrument_pa).value
    header['CD3_3'] = wv_lincal.cdelt1_wavecal.to(default_wv_unit / u.pix).value
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT3'] = default_wv_unit.name

    # define wcs object
    wcs3d = WCS(header)
    if verbose:
        print(f'\n{wcs3d}')

    return wcs3d


def wcs_to_header_using_cd_keywords(wcs3d):
    """Return WCS header using CDi_j keywords.

    This function is a workaround to avoid the problem with the
    astropy.wcs.WCS.to_header() method that replaces CDi_j keywords
    with PCi_j and CDELTi keywords.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.

    Returns
    -------
    header : `~astropy.io.fits.header.Header`
        FITS header with CDi_j keywords.
    """
    header = wcs3d.to_header()

    # Replace PCi_j with CDi_j keywords and remove CDELTi keywords.
    list_of_pc_keys = [key for key in header.keys() if key.startswith('PC')]
    for key in list_of_pc_keys:
        header.rename_keyword(key, f'CD{key[2]}_{key[4]}')
    for key in ['CDELT1', 'CDELT2', 'CDELT3']:
        if key in header:
            del header[key]
    return header


def get_wvparam_from_wcs3d(wcs3d):
    """Return CUNIT1, CRPIX1, CRVAL1, CDELT1 from 3D WCS.

    It is assumed that the wavelength axis corresponds to NAXIS3.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.

    Returns
    -------
    wv_cunit1 : `~astropy.units.core.Unit`
        Default wavelength unit to be employed in CRVAL1 and CDELT1.
    wv_crpix1 :`~astropy.units.Quantity`
        CRPIX1 value.
    wv_crval1 :`~astropy.units.Quantity`
        CRVAL1 value.
    wv_cdelt1 :`~astropy.units.Quantity`
        CDELT1 value.
    """

    wv_cunit1 = wcs3d.wcs.cunit[2]
    wv_crpix1 = wcs3d.wcs.crpix[2] * u.pix
    wv_crval1 = wcs3d.wcs.crval[2] * wv_cunit1

    # Note that the use of wcs3d.wcs.cdelt1[2] raises a RuntimeWarning:
    # "cdelt will be ignored since cd is present"
    # For that reason we read cdelt1 from the cd matrix
    wv_cdelt1 = wcs3d.wcs.cd[2, 2] * wv_cunit1 / u.pix

    return wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1
