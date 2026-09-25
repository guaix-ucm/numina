#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
from astropy.wcs import find_all_wcs
from astropy.wcs import WCS
import logging
import sys


def get_hdu_from_hdul(hdul, extname=None, extnum=None):
    """Get HDU from HDUList.

    The HDU can be specified either by extension name or extension number.
    If both are provided, an error is raised. If neither is provided,
    the first extension (0) is used by default.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        HDUList object containing the FITS file data.
    extname : str, optional
        Extension name for image in input files. Default is None.
    extnum : int, optional
        Extension number for image in input files. Default is None.

    Returns
    -------
    hdu : astropy.io.fits.ImageHDU or astropy.io.fits.PrimaryHDU
        HDU object corresponding to the specified extension.
    """
    logger = logging.getLogger(__name__)

    # Check if both extnum and extname are properly provided
    if extnum is None and extname is None:
        extnum = 0
        if len(hdul) > 1:
            logger.warning(f"Multiple extensions found in {hdul.filename()}.")
            logger.warning(f"Using extension number {extnum} by default.")
    elif extnum is not None and extname is not None:
        logger.error("Please provide either 'extnum' or 'extname', not both.")
        sys.exit(1)

    # Determine the extension to use based on extnum or extname
    if extname is not None:
        extname_image = str(extname).upper()
        if extname_image not in hdul:
            logger.error(f"Extension name '{extname_image}' not found in '{hdul.filename()}'.")
            sys.exit(1)
        extnum_image = hdul.index_of(extname_image)
        logger.info(f"Using extension name '{extname_image}' with number {extnum_image}.")
    else:
        if extnum < 0 or extnum > len(hdul) - 1:
            raise ValueError(f"Extension number {extnum} is out of range for the HDUList.")
        # If extnum is provided, use it to get the extension name
        extname_image = hdul[extnum].name
        logger.info(f"Using extension number {extnum} with name '{extname_image}'.")

    # Get the HDU corresponding to the specified extension
    hdu = hdul[extname_image]

    return hdu


def get_wcs_from_hdu(hdu, wcskey=None):
    """Get WCS from HDU

    The function checks for multiple WCS in the header. If multiple WCS are found,
    the user must specify which one to use via the `wcskey` parameter.
    If only one WCS is found, it is used directly.

    Parameters
    ----------
    hdu : astropy.io.fits.ImageHDU or astropy.io.fits.PrimaryHDU
        HDU object containing the FITS file data.
    wcskey : str, optional
        WCS key to use when multiple WCS are present in the FITS header. Default is None.

    Returns
    -------
    wcs : astropy.wcs.WCS
        WCS object corresponding to the specified extension and WCS key.
    """
    logger = logging.getLogger(__name__)

    # Check if the HDU is an image HDU
    if not isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)):
        logger.error("The specified extension is not an image HDU.")
        sys.exit(1)

    # Check for multiple WCS in the header and handle accordingly
    list_wcs = find_all_wcs(hdu.header)
    if len(list_wcs) > 1:
        logger.warning(f"Number of WCS found: {len(list_wcs)}")
        for w in list_wcs:
            name = w.wcs.name or ""
            logger.info(f"key='{w.wcs.alt}'")
            logger.info(f"  WCSNAME='{name}'")
            logger.info(f"  CTYPE={list(w.wcs.ctype)}")
        if wcskey is None:
            logger.error(f"Multiple WCS found in extension '{hdu.name}'.")
            logger.error(f"Please specify which one to use with 'wcskey'.")
            sys.exit(1)
        else:
            logger.info(f"Using WCS with key '{wcskey}'")
            if wcskey not in [w.wcs.alt for w in list_wcs]:
                raise ValueError(f"WCS key '{wcskey}' not found in extension '{hdu.name}'.")
        wcs = WCS(hdu.header, key=wcskey)
    else:
        wcs = WCS(hdu.header)

    return wcs
