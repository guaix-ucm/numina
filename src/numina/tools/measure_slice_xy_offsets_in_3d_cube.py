#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Determine (X,Y) offsets between slices along NAXIS3"""

import argparse
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate2d
from skimage.registration import phase_cross_correlation
import sys

from numina.array.display.polfit_residuals import polfit_residuals_with_sigma_rejection
import numina.array.imsurfit as imsurfit
from numina.array.imsurfit import vertex_of_quadratic
from numina.array.rescale_array_z1z2 import rescale_array_to_z1z2
import numina.array.utils as utils
from .compare_adr_extensions_in_3d_cube import compare_adr_extensions_in_3d_cube
from .ctext import ctext


def compute_i1i2(i, naxis3, binning_naxis3):
    """Auxiliary function to compute indices for binned slice

    Parameters
    ----------
    i : int
        Array index around which the desired interval [i1,i2] will be
        calculated.
    naxis3 : int
        Maximum allowed value.
    binning_naxis3 : float
        Bin width.

    Returns
    -------
    i1 : int
        Lower array index.
    i2 : int
        Upper array index.
    """

    i1 = int(i - binning_naxis3 / 2)
    if i1 < 0:
        i1 = 0
    i2 = int(i1 + binning_naxis3)
    if i2 > naxis3:
        i2 = naxis3
    return i1, i2


def measure_slice_xy_offsets_in_3d_cube(
        data3d,
        npoints_along_naxis3=100,
        polydeg=2,
        times_sigma_reject=3.0,
        islice_reference=None,
        iterate=True,
        method=1,
        plots=False,
        verbose=False
):
    """Compute (X, Y) offsets of the slices in a 3D datacube

    We are assuming that X correspond to NAXIS1, and Y to NAXIS2,
    where:

    NAXIS3, NAXIS2, NAXIS1 = data3d.shape

    Parameters
    ----------
    data3d : `~numpy.ndarray`
        Data cube.
    npoints_along_naxis3 : int
        Number of equidistant points along NAXIS3 for which the offsets
        in (X,Y) are calculated. In practice, a binning is performed
        along NAXIS3, where the number of slices to be summed in each
        bin is NAXIS3 / npoints_along_naxis3.
    polydeg : int
        Polynomial degree to fit distortion along NAXIS3.
    times_sigma_reject : float
        Times sigma to reject measured offsets when fitting the
        polynomial.
    islice_reference: int or None
        Pixel along NAXIS3 number to derive the initial reference
        binned slice. If None, the central location along NAXIS3 is
        employed. It must be a number from 1 to NAXIS3.
    iterate : bool
        If it's True, the procedure is repeated twice. In the first
        iteration, the binned slice centered at islice_refence is
        used as the reference slice. In the second iteration, the
        reference slice is obtained by averaging the entire corrected
        cube along NAXIS3.
    method : int
        Two methods are implemented:
        - method 1: using skimage.registration.phase_cross_correlation
        - method 2: using scipy.signal.correlated2d
        Both methods seem to yield similar results. Method 1 is faster
        and exhibits smaller residuals than method 2.
    plots : bool
        If True, plot intermediate results.
    verbose : bool
        If True, display intermediate information.

    Returns
    -------
    delta_x_array : `~numpy.ndarray`
        Offsets in X direction (pixels), evaluated along NAXIS3.
    delta_y_array : `~numpy.ndarray`
        Ofssets in Y direction (pixels), evaluated along NAXIS3.
    i1_ref : int
        First slice to derive the reference slice. The slice interval
        corresponds to [i1_ref:i2_ref] following the FITS convention.
    i2_ref : int
        Last slice to derive the reference slice. The slice interval
        corresponds to [i1_ref:i2_ref] following the FITS convention.

    """

    naxis3, naxis2, naxis1 = data3d.shape

    if npoints_along_naxis3 > naxis3:
        if verbose:
            print(f'WARNING: {npoints_along_naxis3=} > {naxis3=}')
            print('         forcing npoints_along_naxis3 = naxis3')
        npoints_along_naxis3 = naxis3

    # reference slice for the first iteration
    if islice_reference is None:
        islice_reference = naxis3 // 2
        if verbose:
            print(f'WARNING: setting {islice_reference=}')

    if islice_reference < 1:
        raise ValueError(f'Unexpected {islice_reference=} < 1')
    if islice_reference > naxis3:
        raise ValueError(f'Unexpected {islice_reference=} > {naxis3=}')

    # bin width along NAXIS3
    binning_naxis3 = naxis3 / npoints_along_naxis3
    if binning_naxis3 == 1:
        islice_array = np.arange(naxis3)
    else:
        islice_array = np.linspace(binning_naxis3 / 2, naxis3 - binning_naxis3 / 2, npoints_along_naxis3).astype(int)
    if verbose:
        print(f'{binning_naxis3=}')

    niterations = 1
    if iterate:
        niterations += 1

    # duplicate the input 3D array
    data3d_work = data3d.copy()

    # main loop in number of iteration
    delta_x_array = None
    delta_y_array = None
    i1_ref = None
    i2_ref = None
    for iteration in range(niterations):
        # compute reference binned slice
        if iteration == 0:
            i1_ref, i2_ref = compute_i1i2(islice_reference-1, naxis3, binning_naxis3)
            slice_reference = np.mean(data3d_work[i1_ref:i2_ref, :, :], axis=0)
        else:
            data3d_corrected = np.zeros_like(data3d_work)
            for islice in np.arange(naxis3):
                data3d_corrected[islice, :, :] = shift(
                    data3d_work[islice, :, :],
                    (-delta_y_array[islice], -delta_x_array[islice]),
                    mode='constant',
                    cval=0
                )
            i1_ref = 0
            i2_ref = naxis3
            slice_reference = np.mean(data3d_corrected[i1_ref:i2_ref, :, :], axis=0)
            # del data3d_corrected
        if verbose:
            # FITS convention
            print(f'iteration: {iteration} --> reference slice in [{i1_ref+1}:[{i2_ref}]')
        if plots:
            fig, ax = plt.subplots()
            ax.imshow(slice_reference, origin='lower')
            ax.set_xlabel('Array index (NAXIS1 direction)')
            ax.set_ylabel('Array index (NAXIS2 direction)')
            # FITS convention
            ax.set_title(f'slice_reference along NAXIS3\n(i1_ref: {i1_ref+1}, i2_ref: {i2_ref}, iteration {iteration})')
            plt.tight_layout()
            plt.show()

        # rescale image to (0, 1)
        slice_reference, _ = rescale_array_to_z1z2(slice_reference, (0, 1))

        # loop in number of binned slice
        xfit = []
        delta_x_slice = []
        delta_y_slice = []
        for islice in islice_array:
            i1, i2 = compute_i1i2(islice, naxis3, binning_naxis3)
            xfit.append((i1 + i2 - 1) / 2)
            slice_binned = np.nanmean(data3d_work[i1:i2, :, :], axis=0)
            slice_binned, _ = rescale_array_to_z1z2(slice_binned, (0, 1))
            if method == 1:
                yx_offsets, _, _ = phase_cross_correlation(
                    reference_image=slice_reference,
                    moving_image=slice_binned,
                    upsample_factor=100,
                    overlap_ratio=0.90
                )
            elif method == 2:
                corr_self = correlate2d(
                    in1=slice_reference,
                    in2=slice_reference,
                    mode='full',
                    boundary='fill',
                    fillvalue=0
                )
                corr = correlate2d(
                    in1=slice_reference,
                    in2=slice_binned,
                    mode='full',
                    boundary='fill',
                    fillvalue=0
                )
                maxindex_self = np.unravel_index(np.argmax(corr_self), corr_self.shape)
                maxindex = np.unravel_index(np.argmax(corr), corr.shape)
                refine_box = 3
                region_refine_self = utils.image_box(
                    maxindex_self,
                    corr_self.shape,
                    box=(refine_box, refine_box)
                )

                region_refine = utils.image_box(
                    maxindex,
                    corr.shape,
                    box=(refine_box, refine_box)
                )
                coeffs_self, = imsurfit.imsurfit(corr_self[region_refine_self], order=2)
                coeffs, = imsurfit.imsurfit(corr[region_refine], order=2)
                xm, ym = vertex_of_quadratic(coeffs_self)
                maxindex_self += np.asarray([ym, xm])
                xm, ym = vertex_of_quadratic(coeffs)
                maxindex += np.asarray([ym, xm])
                yx_offsets = np.asarray(maxindex) - np.asarray(maxindex_self)
            else:
                raise ValueError(f'Unexpected {method=}')
            delta_x_slice.append(-yx_offsets[1])
            delta_y_slice.append(-yx_offsets[0])
        xfit = np.array(xfit)
        delta_x_slice = np.array(delta_x_slice)
        delta_y_slice = np.array(delta_y_slice)
        for yfit, title in zip(
                [delta_x_slice, delta_y_slice],
                ['delta_x_slice', 'delta_y_slice']
        ):
            if plots:
                debugplot = 1
                fig = plt.figure()
            else:
                debugplot = 0
                fig = None
            poly, residuals, reject = polfit_residuals_with_sigma_rejection(
                x=xfit,
                y=yfit,
                deg=polydeg,
                times_sigma_reject=times_sigma_reject,
                title=title,
                debugplot=debugplot,
                fig=fig
            )
            if verbose:
                print(f'{title} -> {poly=}')
            if plots:
                plt.tight_layout()
                plt.show()
            result = poly(np.arange(naxis3))
            if title == 'delta_x_slice':
                delta_x_array = result
            else:
                delta_y_array = result

    return delta_x_array, delta_y_array, i1_ref+1, i2_ref   # FITS convention


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(description="Determine (X,Y) offsets between slices along NAXIS3")
    parser.add_argument("filename", help="Input 3D FITS file")
    parser.add_argument("npoints", help="Number of points along NAXIS3", type=int)
    parser.add_argument("--extname", help="Output extension name to store result (default None)",
                        type=str, default='None')
    parser.add_argument("--polydeg", help="Polynomial degree to fit distortion", type=int, default=2)
    parser.add_argument("--times_sigma_reject",
                        help="Times sigma to reject fitted offsets (default 3.0)",
                        type=float, default=3.0)
    parser.add_argument("--islice_reference",
                        help="Initial pixel corresponding to the reference slice (from 1 to NAXIS3)",
                        type=int)
    parser.add_argument("--iterate", help="Force one iteration", action="store_true")
    parser.add_argument("--method", help="Method (1: skimage, 2: scipy)",
                        type=int, choices=[1, 2], default=1)
    parser.add_argument("--plots", help="Plot intermediate results", action="store_true")
    parser.add_argument("--verbose", help="Display intermediate information", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # protections
    extname = args.extname.upper()
    if len(extname) > 8:
        raise ValueError(f"Extension '{extname}' must be less than 9 characters")

    with fits.open(args.filename) as hdul:
        primary_header = hdul[0].header
        data3d = hdul[0].data

    if primary_header['NAXIS'] != 3:
        raise ValueError(f"Expected NAXIS=3 not found in PRIMARY HDU")

    delta_x_array, delta_y_array, i1_ref, i2_ref = measure_slice_xy_offsets_in_3d_cube(
        data3d=data3d,
        npoints_along_naxis3=args.npoints,
        polydeg=args.polydeg,
        times_sigma_reject=args.times_sigma_reject,
        islice_reference=args.islice_reference,
        iterate=args.iterate,
        method=args.method,
        plots=args.plots,
        verbose=args.verbose
    )

    # save result in extension
    if extname != 'NONE':
        if args.verbose:
            print(f'Updating file {args.filename}')
        # binary table to store result
        col1 = fits.Column(name='Delta_x', format='D', array=delta_x_array, unit='pixel')
        col2 = fits.Column(name='Delta_y', format='D', array=delta_y_array, unit='pixel')
        hdu_result = fits.BinTableHDU.from_columns([col1, col2])
        hdu_result.name = extname.upper()
        hdu_result.header['METHOD'] = (args.method, '1: skimage, 2: scipy')
        hdu_result.header['POLYDEG'] = (args.polydeg, 'Polynomial degree to fit distortion')
        hdu_result.header['TSIGMA'] = (args.times_sigma_reject, 'Times sigma to reject fitted offsets')
        hdu_result.header['NP_FIT'] = (args.npoints, 'Number of points along NAXIS3')
        hdu_result.header['ITERATE'] = (args.iterate, 'Iterate procedure (T: True, F: False)')
        if args.islice_reference is not None:
            i0_ref = args.islice_reference
        else:
            i0_ref = -1
        hdu_result.header['I0_REF'] = (i0_ref, 'Initial pixel for reference slice (-1: None)')
        hdu_result.header['I1_REF'] = (i1_ref, 'First pixel of reference slice')
        hdu_result.header['I2_REF'] = (i2_ref, 'Last pixel of reference slice')
        wcs3d = WCS(primary_header)
        naxis1, naxis2, naxis3 = wcs3d.pixel_shape
        wave = wcs3d.spectral.pixel_to_world(np.arange(naxis3))
        reference_vacuum_wavelength = (wave[i1_ref-1] + wave[i2_ref-1]) / 2
        hdu_result.header['REFEWAVE'] = (reference_vacuum_wavelength.to(u.m).value,
                                         'Reference vacuum wavelength (m)')
        # open and update existing FITS file
        hdul = fits.open(args.filename, mode='update')
        if extname in hdul:
            if args.verbose:
                print(f"Updating extension '{extname}'")
            hdul[extname] = hdu_result
        else:
            if args.verbose:
                print(f"Adding new extension '{extname}'")
            hdul.append(hdu_result)
        hdul.flush()
        hdul.close()

    # display results
    if args.plots:
        suptitle = f"file: {args.filename} (extension: {extname})\nreference slices in [{i1_ref}:{i2_ref}]"
        compare_adr_extensions_in_3d_cube(args.filename, extname1=extname, extname2=None, suptitle=suptitle)


if __name__ == "__main__":

    main()
