#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
import numpy as np


def save_image2d_detector_method0(
        header_keys,
        image2d_detector_method0,
        prefix_intermediate_fits,
        bitpix
):
    """Save the two 2D images: RSS and detector.

    Parameters
    ----------
    header_keys : `~astropy.io.fits.header.Header`
        FITS header with additional keywords to be merged together with
        the default keywords.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    bitpix : int, optional
        BITPIX value for the FITS file.
    """

    if len(prefix_intermediate_fits) > 0:
        # --------------------------------------
        # spectroscopic 2D image in the detector
        # --------------------------------------
        if bitpix == 16:
            # avoid overflow
            image2d_detector_method0[image2d_detector_method0 > 65535] = 65535
            # round to integer and save as BITPIX=16 (unsigned short)
            hdu = fits.PrimaryHDU(np.round(image2d_detector_method0).astype(np.uint16))
        else:
            raise ValueError(f'Unsupported BITPIX value: {bitpix}')
        pos0 = len(hdu.header) - 1
        hdu.header.update(header_keys)
        hdu.header.insert(
            pos0, ('COMMENT', "FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        hdu.header.insert(
            pos0 + 1, ('COMMENT', "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_detector_2D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')
