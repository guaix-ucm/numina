#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
import logging

from astropy import wcs
from astropy.io import fits
import astropy.units as u
import numpy as np
from pathlib import Path

from .define_3d_wcs import get_wvparam_from_wcs3d
from .define_3d_wcs import wcs_to_header_using_cd_keywords


def save_image2d_rss(
    wcs3d, header_keys, image2d_rss, method, prefix_intermediate_fits, bitpix, logger=None, output_dir="."
):
    """Save the RSS image.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    header_keys : `~astropy.io.fits.header.Header`
        FITS header with additional keywords to be merged together with
        the WCS information.
    image2d_rss : `~numpy.ndarray`
        2D array containing the RSS image.
    method : int
        Integer indicating the method: 0 or 1.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    bitpix : int
        BITPIX value for the FITS file.
    logger : `~logging.Logger`, optional
        Logger for logging messages. If None, the root logger is used.
    output_dir : str or `~pathlib.Path`, optional
        Output directory to store results. Default is the current directory.
    """
    if logger is None:
        logger = logging.getLogger()

    if len(prefix_intermediate_fits) > 0:
        # ------------------------------------------------
        # 1) spectroscopic 2D image with contiguous slices
        # ------------------------------------------------
        # ToDo: compute properly the parameters corresponding to the spatial axis
        # Note that using: wcs2d = wcs3d.sub(axes=[0, 1])
        # selecting the 1D spectral and one of the 1D spatial info of the 3D WCS
        # does not work:
        # "astropy.wcs._wcs.InconsistentAxisTypesError: ERROR 4 in wcs_types()
        #  Unmatched celestial axes."
        # For that reason we try a different approach:
        wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)
        wcs2d = wcs.WCS(naxis=2)
        wcs2d.wcs.crpix = [wv_crpix1.value, 1]  # reference pixel coordinate
        wcs2d.wcs.crval = [wv_crval1.value, 0]  # world coordinate at reference pixel
        wcs2d.wcs.cdelt = [wv_cdelt1.value, 1]
        wcs2d.wcs.ctype = ["WAVE", ""]  # ToDo: fix this
        wcs2d.wcs.cunit = [wv_cunit1, u.pix]
        outfile = f"{prefix_intermediate_fits}_rss_2D_method{method}.fits"
        if bitpix == 16:
            if image2d_rss.max() <= 65535:
                if image2d_rss.min() < 0:
                    raise ValueError(
                        f"Negative values found in {outfile} but BITPIX=16 does not support negative values."
                    )
                hdu = fits.PrimaryHDU(np.round(image2d_rss).astype(np.uint16))
                bitpix_used = 16
            else:
                # use float to avoid saturation problem
                logger.warning(f"The maximum value in {outfile} is greater than 65535.")
                logger.warning("Saving the image using float32 to avoid saturation.")
                hdu = fits.PrimaryHDU(image2d_rss.astype(np.float32))
                bitpix_used = -32
        elif bitpix == -32:
            hdu = fits.PrimaryHDU(image2d_rss.astype(np.float32))
            bitpix_used = -32
        else:
            raise ValueError(f"Unsupported BITPIX value: {bitpix}")
        pos0 = len(hdu.header) - 1
        hdu.header.extend(wcs_to_header_using_cd_keywords(wcs2d), update=True)
        hdu.header.update(header_keys)
        hdu.header.insert(pos0, ("COMMENT", "FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        hdu.header.insert(
            pos0 + 1, ("COMMENT", "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H")
        )
        hdul = fits.HDUList([hdu])
        logger.info(f"Saving file: {outfile} (BITPIX={bitpix_used})")
        hdul.writeto(f"{Path(output_dir) / outfile}", overwrite=True)
