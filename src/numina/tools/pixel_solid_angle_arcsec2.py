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
from argparse import RawTextHelpFormatter
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from tqdm import tqdm
from rich import print
from scipy.ndimage import median_filter
from spherical_geometry.polygon import SphericalPolygon
import sys

from .add_script_info_to_fits_history import add_script_info_to_fits_history


def pixel_solid_angle_arcsec2(wcs, naxis1, naxis2, method=3, kernel_size=None, verbose=False):
    """Compute the solid angle (arcsec**2) of every pixel.

    This function computes the solid angle for each pixel in a 2D image.
    When using WCS with a very small projected pixel size (e.g., 0.01 arcsec),
    Method 2 is recommended but slow. Method 1 is slow and does not
    work well for very small pixel sizes, but we keep the code here for completeness.
    Method 3 is fast and works well for most cases, but it may not be accurate
    for very small pixel sizes.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        WCS object defining the celestial coordinates of the image.
    naxis1 : int
        Number of pixels along the first axis (NAXIS1).
    naxis2 : int
        Number of pixels along the second axis (NAXIS2).
    method : int, optional
        Method to compute the solid angle.
        1: Use spherical polygons (slow).
        2: Use spherical polygons with a different approach (slow).
        3: Use spherical coordinates and distances (fast, default).
    kernel_size : int, optional
        Size of the kernel for smoothing the result using a median filter.
        If None, no smoothing is applied.
    verbose : bool, optional
        If True, display intermediate information.

    Returns
    -------
    result : `numpy.ndarray`
        2D array with the solid angle (arcsec**2) for each pixel in
        the input image.
    """

    if verbose:
        print(f"Computing solid angle for {naxis1} x {naxis2} pixels using method {method}.")

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

    if method == 1:
        # Use spherical polygons to compute the solid angle
        result_spherical = result_spherical.reshape(naxis2 + 1, naxis1 + 1)
        result = np.zeros((naxis2, naxis1))
        for i in tqdm(range(naxis2), desc="NAXIS2", disable=not verbose):
            for j in range(naxis1):
                polygon = SphericalPolygon.from_radec(
                    lon=[result_spherical[i, j].ra.rad,
                         result_spherical[i, j + 1].ra.rad,
                         result_spherical[i + 1, j + 1].ra.rad,
                         result_spherical[i + 1, j].ra.rad],
                    lat=[result_spherical[i, j].dec.rad,
                         result_spherical[i, j + 1].dec.rad,
                         result_spherical[i + 1, j + 1].dec.rad,
                         result_spherical[i + 1, j].dec.rad],
                    degrees=False,
                )
                result[i, j] = polygon.area() * (180 / np.pi) ** 2 * 3600 ** 2
    elif method == 2:
        # Use spherical polygons with a different approach to compute the solid angle
        result_spherical = result_spherical.reshape(naxis2 + 1, naxis1 + 1)
        result = np.zeros((naxis2, naxis1))
        for i in tqdm(range(naxis2), desc="NAXIS2", disable=not verbose):
            for j in range(naxis1):
                dist1a = result_spherical[i, j].separation(result_spherical[i, j + 1]).arcsec
                dist1b = result_spherical[i + 1, j].separation(result_spherical[i + 1, j + 1]).arcsec
                dist2a = result_spherical[i, j].separation(result_spherical[i + 1, j]).arcsec
                dist2b = result_spherical[i, j + 1].separation(result_spherical[i + 1, j + 1]).arcsec
                # average distances corresponding to opposite sides of each pixel
                dist1 = (dist1a + dist1b) / 2
                dist2 = (dist2a + dist2b) / 2
                # solid angle (arcsec**2) of the pixel
                result[i, j] = dist1 * dist2
    elif method == 3:
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
        # average distances corresponding to the left and right sides of each pixel
        pixel_size_naxis2 = (result_naxis2[:, :-1] + result_naxis2[:, 1:]) / 2
        # pixel size (arcsec**2)
        result = pixel_size_naxis1 * pixel_size_naxis2
    else:
        raise ValueError(f"Invalid method: {method}. Choose 1, 2, or 3.")

    # smooth result
    if kernel_size is not None:
        if verbose:
            print(f"Applying median filter with kernel size {kernel_size}.")
        result = median_filter(result, size=kernel_size, mode='nearest')

    return result


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Compute the solid angle in arcsec^2 for each pixel in a 2D image.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('input_file', type=str,
                        help='FITS file containing the image data.')
    parser.add_argument('output_file', type=str,
                        help='Output FITS file to save the solid angle data.')
    parser.add_argument("--extname", type=str,
                        help="Extension name of the input HDU (default: 'PRIMARY').",
                        default='PRIMARY')
    parser.add_argument("--method", type=int, default=3,
                        help="Method to compute the solid angle:\n"
                             "1: Use spherical polygons (slow)\n"
                             "   (not recommended for very small pixel sizes)\n"
                             "2: Use spherical polygons (different approach, slow)\n"
                             "   (recommended for small pixel sizes)\n"
                             "3: Use spherical coordinates and distances (fast)\n"
                             "   (default, not recommended for very small pixel sizes)")
    parser.add_argument("--kernel_size", type=int, default=None,
                        help="Kernel size for smoothing the result using a median\n"
                             "filter. If not specified, no smoothing is applied.")
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
            print(f'{arg}: {value}')

    if args.echo:
        print('[bold red]Executing:\n' + ' '.join(sys.argv) + '[/bold red]')

    input_file = args.input_file
    output_file = args.output_file
    extname = args.extname
    kernel_size = args.kernel_size
    method = args.method
    verbose = args.verbose

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

    result = pixel_solid_angle_arcsec2(
        wcs=wcs,
        naxis1=naxis1,
        naxis2=naxis2,
        method=method,
        kernel_size=kernel_size,
        verbose=verbose
    )

    # Create a new FITS HDU with the solid angle data
    solid_angle_hdu = fits.PrimaryHDU(data=result.astype(np.float32))
    header = solid_angle_hdu.header
    add_script_info_to_fits_history(header, args)
    if args.verbose:
        print(f"Saving solid angle data to {output_file}.")
    solid_angle_hdu.writeto(output_file, overwrite=True)
