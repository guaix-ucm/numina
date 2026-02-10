#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
import logging

from astropy.io import fits
import numpy as np

from .define_3d_wcs import wcs_to_header_using_cd_keywords

def generate_image3d_method0_ifu(
        wcs3d,
        header_keys,
        simulated_x_ifu_all,
        simulated_y_ifu_all,
        simulated_wave_all,
        bins_x_ifu,
        bins_y_ifu,
        bins_wave,
        prefix_intermediate_fits,
        logger=None
):
    """Compute 3D image3 IFU, method 0

    The image is calculated by performing a three-dimensional histogram
    over the X, Y coordinates in the IFU and the wavelength of the
    simulated photons.

    The result is saved as a FITS file.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    header_keys : `~astropy.io.fits.header.Header`
        FITS header with additional keywords to be merged together with
        the WCS information.
    simulated_x_ifu_all : `~astropy.units.Quantity`
        Simulated X coordinates of the photons in the IFU.
    simulated_y_ifu_all : `~astropy.units.Quantity`
        Simulated Y coordinates of the photons in the IFU.
    simulated_wave_all : `~astropy.units.Quantity`
        Simulated wavelengths of the photons in the IFU.
    bins_x_ifu : `~numpy.ndarray`
        Bin edges in the naxis1_ifu direction
        (along the slice).
    bins_y_ifu : `~numpy.ndarray`
        Bin edges in the naxis2_ifu direction
        (perpendicular to the slice).
    bins_wave : `~numpy.ndarray`
        Bin edges in the wavelength direction.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    logger : `~logging.Logger`, optional
        Logger for logging messages. If `None`, the root logger is used.
    """
    if logger is None:
        logger = logging.getLogger()

    # generate image
    image3d_method0_ifu, edges = np.histogramdd(
        sample=(simulated_wave_all.value, simulated_y_ifu_all.value, simulated_x_ifu_all.value),
        bins=(bins_wave.value, bins_y_ifu.value, bins_x_ifu.value)
    )

    # save FITS file
    if len(prefix_intermediate_fits) > 0:
        hdu = fits.PrimaryHDU(image3d_method0_ifu.astype(np.uint16))
        pos0 = len(hdu.header) - 1
        hdu.header.extend(wcs_to_header_using_cd_keywords(wcs3d), update=True)
        hdu.header.update(header_keys)
        hdu.header.insert(
            pos0, ('COMMENT', "FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        hdu.header.insert(
            pos0 + 1, ('COMMENT', "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_ifu_3D_method0.fits'
        logger.info(f'Saving file: {outfile}')
        hdul.writeto(f'{outfile}', overwrite='yes')
