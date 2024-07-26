#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Determine (X,Y) offsets between slices along NAXIS3"""

import argparse
from astropy.io import fits
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


def compute_slice_xy_offsets_in_3d_cube(
        data3d,
        npoints_along_naxis3=100,
        polydeg=2,
        islice_reference=None,
        iterate=True,
        method=1,
        plots=False,
        debug=False
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
        in (X,Y) are calculated. Practically, a binning is performed
        along NAXIS3, where the number of slices to be summed in each
        bin is NAXIS3 / npoints_along_naxis3.
    polydeg : int
        Polynomial degree to fit distortion along NAXIS3.
    islice_reference: int
        Slice number (array index) to derive the initial reference
        binned slice. If None, the central location along NAXIS3 is
        employed.
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
    debug : bool
        If True, display intermediate information.

    Returns
    -------
    delta_x_array : `~numpy.ndarray`
        Offsets in X direction (pixels), evaluated along NAXIS3.
    delta_y_array : `~numpy.ndarray`
        Ofssets in Y direction (pixels), evaluated along NAXIS3.

    """

    naxis3, naxis2, naxis1 = data3d.shape

    if npoints_along_naxis3 > naxis3:
        if debug:
            print(f'WARNING: {npoints_along_naxis3=} > {naxis3=}')
            print(f'         forcing npoints_along_naxis3 = naxis3')
        npoints_along_naxis3 = naxis3

    # reference slice for the first iteration
    if islice_reference is None:
        islice_reference = naxis3 // 2
        if debug:
            print(f'WARNING: setting {islice_reference=}')

    if islice_reference < 0:
        raise ValueError(f'Unexpected {islice_reference=} < 0')
    if islice_reference >= naxis3:
        raise ValueError(f'Unexpected {islice_reference=} >= {naxis3=}')

    # bin width along NAXIS3
    binning_naxis3 = naxis3 / npoints_along_naxis3
    if binning_naxis3 == 1:
        islice_array = np.arange(naxis3)
    else:
        islice_array = np.linspace(binning_naxis3 / 2, naxis3 - binning_naxis3 / 2, npoints_along_naxis3).astype(int)
    if debug:
        print(f'{binning_naxis3=}')

    niterations = 1
    if iterate:
        niterations += 1

    # duplicate the input 3D array
    data3d_work = data3d.copy()

    # main loop in number of iteration
    delta_x_array = None
    delta_y_array = None
    for iteration in range(niterations):
        # compute reference binned slice
        if iteration == 0:
            i1_ref, i2_ref = compute_i1i2(islice_reference, naxis3, binning_naxis3)
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
        if debug:
            print(f'iteration: {iteration} --> {i1_ref=}, {i2_ref=}')
        if plots:
            fig, ax = plt.subplots()
            ax.imshow(slice_reference, origin='lower')
            ax.set_xlabel('Array index (NAXIS1 direction)')
            ax.set_ylabel('Array index (NAXIS2 direction)')
            ax.set_title(f'slice_reference\n(i1_ref: {i1_ref}, i2_ref: {i2_ref}, iteration {iteration})')
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
                times_sigma_reject=3.0,
                title=title,
                debugplot=debugplot,
                fig=fig
            )
            if debug:
                print(f'{title} -> {poly=}')
            if plots:
                plt.tight_layout()
                plt.show()
            result = poly(np.arange(naxis3))
            if title == 'delta_x_slice':
                delta_x_array = result
            else:
                delta_y_array = result

    return delta_x_array, delta_y_array


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(description="Determine (X,Y) offsets between slices along NAXIS3")
    parser.add_argument("input", help="Input FITS file")
    parser.add_argument("npoints", help="Number of points along NAXIS3", type=int)
    parser.add_argument("--output", help="Output FITS file")
    parser.add_argument("--polydeg", help="Polynomial degree to fit distortion", type=int, default=2)
    parser.add_argument("--islice_reference", help="Array index corresponding to the reference slice", type=int)
    parser.add_argument("--iterate", help="Force one iteration", action="store_true")
    parser.add_argument("--method", help="Method (1: skimage, 2: scipy)", type=int, choices=[1, 2], default=1)
    parser.add_argument("--plots", help="Plot intermediate results", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    parser.add_argument("--debug", help="Debug", action="store_true")

    args = parser.parse_args(args=args)
    if args.debug:
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    with fits.open(args.input) as hdul:
        data3d = hdul[0].data

    delta_x_array, delta_y_array = compute_slice_xy_offsets_in_3d_cube(
        data3d=data3d,
        npoints_along_naxis3=args.npoints,
        polydeg=args.polydeg,
        islice_reference=args.islice_reference,
        iterate=args.iterate,
        method=args.method,
        plots=args.plots,
        debug=args.debug
    )

    naxis3, naxis2, axis1 = data3d.shape

    if args.output is not None:
        # save result as a FITS file with a 2D array:
        # first row: delta_x_array
        # second row: delta_y_array
        arrayout = np.zeros((2, naxis3))
        arrayout[0, :] = delta_x_array
        arrayout[1, :] = delta_y_array
        hdu = fits.PrimaryHDU(arrayout.astype(np.float32))
        hdul = fits.HDUList([hdu])
        if args.debug:
            print(f'Saving file {args.output}')
        hdul.writeto(args.output, overwrite='yes')

    if args.plots:
        fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 6.4))
        xplot = np.arange(naxis3)
        for iplot, yplot, label in zip(range(2), [delta_x_array, delta_y_array], 'xy'):
            ax = axarr[iplot]
            ax.plot(xplot, yplot, '.')
            ax.axhline(0, linestyle='--', color='grey')
            ax.set_xlabel('Array index (along NAXIS3)')
            ax.set_ylabel(f'delta_{label}_array (pixels)')
        plt.suptitle(args.input)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    main()
