#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute the solid angle in arcsec^2 for each pixel in a 2D image."""

import argparse
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy.ndimage import median_filter
import sys

from .ctext import ctext


def pixel_solid_angle_arcsec2(wcs, naxis1, naxis2, kernel_size=None):
    """Compute the solid angle (arcsec**2) of every pixel.
    
    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        WCS object defining the celestial coordinates of the image.
    naxis1 : int
        Number of pixels along the first axis (NAXIS1).
    naxis2 : int
        Number of pixels along the second axis (NAXIS2).
    kernel_size : int, optional
        Size of the kernel for smoothing the result using a median filter.
        If None, no smoothing is applied.

    Returns
    -------
    result : `numpy.ndarray`
        2D array with the solid angle (arcsec**2) for each pixel in
        the input image.
    """

    # X, Y coordinates (2D image array, following the FITS criterium)
    # corresponding to the four corners of all the image pixels
    borders_x = np.arange(naxis1 + 1) + 0.5
    borders_y = np.arange(naxis2 + 1) + 0.5
    meshgrid = np.meshgrid(borders_x, borders_y)
    ix_array = meshgrid[0].flatten()
    iy_array = meshgrid[1].flatten()

    # spherical coordinates of the four corners of all the image pixels
    result_spherical = SkyCoord.from_pixel(
        xp=ix_array,
        yp=iy_array,
        wcs=wcs,
        origin=1,
        mode='all'
    )

    # cartesian coordinates of the four corners of all the image pixels
    x = result_spherical.cartesian.x.value.reshape(naxis2 + 1, naxis1 + 1)
    y = result_spherical.cartesian.y.value.reshape(naxis2 + 1, naxis1 + 1)
    z = result_spherical.cartesian.z.value.reshape(naxis2 + 1, naxis1 + 1)

    # dot product of consecutive points along NAXIS1
    dot_product_naxis1 = x[:, :-1] * x[:, 1:] + \
        y[:, :-1] * y[:, 1:] + z[:, :-1] * z[:, 1:]
    # distance (arcsec) between consecutive points along NAXIS1
    result_naxis1 = np.arccos(dot_product_naxis1) * 180 / np.pi * 3600
    # average distances corresponding to the upper and lower sides of each pixel
    pixel_size_naxis1 = (result_naxis1[:-1, :] + result_naxis1[1:, :]) / 2

    # dot product of consecutive points along NAXIS2
    dot_product_naxis2 = x[:-1, :] * x[1:, :] + \
        y[:-1, :] * y[1:, :] + z[:-1, :] * z[1:, :]
    # distance (arcsec) between consecutive points along NAXIS2
    result_naxis2 = np.arccos(dot_product_naxis2) * 180 / np.pi * 3600
    # averange distances corresponding to the left and right sides of each pixel
    pixel_size_naxis2 = (result_naxis2[:, :-1] + result_naxis2[:, 1:]) / 2

    # pixel size (arcsec**2)
    result = pixel_size_naxis1 * pixel_size_naxis2

    # smooth result
    if kernel_size is not None:
        result = median_filter(result, size=kernel_size, mode='nearest')

    return result


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Compute the solid angle in arcsec^2 for each pixel in a 2D image."
    )
    parser.add_argument('input_file', type=str, 
                        help='FITS file containing the image data.')
    parser.add_argument('output_file', type=str,
                        help='Output FITS file to save the solid angle data.')
    parser.add_argument("--extname", type=str, 
                        help="Extension name of the input HDU (default: 'PRIMARY').", 
                        default='PRIMARY')
    parser.add_argument("--kernel_size", type=int, default=None,
                        help="Size of the kernel for smoothing the result using a median filter. "
                             "If not specified, no smoothing is applied.")
    parser.add_argument("--verbose",
                        help="Display intermediate information",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    input_file = args.input_file
    output_file = args.output_file
    extname = args.extname
    kernel_size = args.kernel_size

    with fits.open(input_file) as hdul:
        if extname not in hdul:
            raise ValueError(f"Extension '{extname}' not found in {input_file}.")
        hdu_image = hdul[extname]
        naxis = hdu_image.header['NAXIS']
        if naxis == 2:
            wcs = WCS(hdu_image.header)
        elif naxis == 3:
            wcs = WCS(hdu_image.header).celestial
        else:
            raise ValueError(f"Unsupported NAXIS value: {naxis}. Expected 2 or 3.")
        naxis1 = hdu_image.header['NAXIS1']
        naxis2 = hdu_image.header['NAXIS2']
        if args.verbose:
            print(f"{naxis1=}")
            print(f"{naxis2=}")
            print(f"Celestial coordinates WCS:\n{wcs}")

    result = pixel_solid_angle_arcsec2(wcs, naxis1, naxis2, kernel_size)

    solid_angle_hdu = fits.PrimaryHDU(data=result.astype(np.float32))
    header = solid_angle_hdu.header
    header['HISTORY'] = 'Solid angle in arcsec^2 for each pixel.'

    if args.verbose:
        print(f"Saving solid angle data to {output_file}.")
    solid_angle_hdu.writeto(output_file, overwrite=True)
