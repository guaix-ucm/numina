#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Extract 2D slice from 3D FITS file."""

from astropy.io import fits
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from rich import print
from rich_argparse import RichHelpFormatter
import sys


def extract_slice(input, axis, i1, i2, method, wavecal, transpose, vmin, vmax, noplot, output, png=None):
    """Extract 2D slice.

    Parameters
    ----------
    input : str
        Input FITS file name.
    axis : int
        Axis to be collapsed in output.
    i1 : int
        First pixel of the projected axis (FITS criterum).
    i2 : int
        Last pixel of the projected axis (FITS criterium).
    method : str
        Collapse method. Choices are "sum", "mean", "median"
    wavecal : str
        Wavelength calibration type. Choices are:
        - "none": ignore wavelength calibration parameters
        - "fridasimulator": the CDELT3 value is 1.0 and the
          wavelength increment must be read from PC3_3
        - "1", "2" or "3": read the traditional CTYPE?, CRPIX?, CRVAL?,
          and CDELT?, with ? = 1, 2 or 3
    transpose : bool
        If True, transpose data array in output.
    vmin : float or None
        Minimum data range that the colormap covers.
    vmax : float or None
        Maximum data range that the colormap covers.
    noplot : bool
        If True, skip plotting of the result.
    output : str
        Output FITS file name.
    png : str or None
        Output PNG file name (plot of the result).

    """

    # protections
    if not (1 <= axis <= 3):
        raise ValueError(f'{axis=} out of valid range (1, 2 or 3)')

    # read first FITS file
    with fits.open(input) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data.astype(float)
    naxis = header['naxis']
    if naxis != 3:
        raise ValueError(f'Unexpected input {naxis=} (it must be 3)')
    naxis3 = header[f'naxis{axis}']

    # set last pixel to maximum value when i2==0
    if i2 == 0:
        i2 = naxis3

    if not (1 <= i1 <= naxis3):
        raise ValueError(f'Unexpected {i1=}. Must be a number in [1,...,{naxis3}]')
    if not (1 <= i2 <= naxis3):
        raise ValueError(f'Unexpected {i2=}. Must be a number in [1,...,{naxis3}]')
    if i2 < i1:
        raise ValueError(f'{i1=} must be <= {i2=}')

    if axis == 1:
        if np.any(np.isnan(data[:, :, (i1 - 1):i2])):
            print(f'Warning: NaN values found in input data for axis {axis} '
                  f'between pixels {i1} and {i2}. They will be ignored.')
        if method == "sum":
            slice2d = np.nansum(data[:, :, (i1 - 1):i2], axis=2, keepdims=False)
        elif method == "mean":
            slice2d = np.nanmean(data[:, :, (i1 - 1):i2], axis=2, keepdims=False)
        elif method == "median":
            slice2d = np.nanmedian(data[:, :, (i1 - 1):i2], axis=2, keepdims=False)
        else:
            raise ValueError(f'Unexpected {method=}')
    elif axis == 2:
        if np.any(np.isnan(data[:, (i1 - 1):i2, :])):
            print(f'Warning: NaN values found in input data for axis {axis} '
                  f'between pixels {i1} and {i2}. They will be ignored.')
        if method == "sum":
            slice2d = np.nansum(data[:, (i1 - 1):i2, :], axis=1, keepdims=False)
        elif method == "mean":
            slice2d = np.nanmean(data[:, (i1 - 1):i2, :], axis=1, keepdims=False)
        elif method == "median":
            slice2d = np.nanmedian(data[:, (i1 - 1):i2, :], axis=1, keepdims=False)
        else:
            raise ValueError(f'Unexpected {method=}')
    else:
        if np.any(np.isnan(data[(i1 - 1):i2, :, :])):
            print(f'Warning: NaN values found in input data for axis {axis} '
                  f'between pixels {i1} and {i2}. They will be ignored.')
        if method == "sum":
            slice2d = np.nansum(data[(i1 - 1):i2, :, :], axis=0, keepdims=False)
        elif method == "mean":
            slice2d = np.nanmean(data[(i1 - 1):i2, :, :], axis=0, keepdims=False)
        elif method == "median":
            slice2d = np.nanmedian(data[(i1 - 1):i2, :, :], axis=0, keepdims=False)
        else:
            raise ValueError(f'Unexpected {method=}')

    if transpose:
        slice2d = np.transpose(slice2d)

    # plot result
    if not noplot:
        naxis2, naxis1 = slice2d.shape
        if wavecal != 'none':
            aspect = 'auto'
        else:
            aspect = None
        xmin, xmax = 0.5, naxis1 + 0.5
        ymin, ymax = 0.5, naxis2 + 0.5
        extent = [xmin, xmax, ymin, ymax]
        fig, ax = plt.subplots()
        img = ax.imshow(slice2d, origin='lower', extent=extent, vmin=vmin, vmax=vmax,
                        aspect=aspect, interpolation='None')
        ax.set_xlabel('pixel (output NAXIS1)')
        ax.set_ylabel('pixel (output NAXIS2)')
        title = f'{input}\n(collapsed input NAXIS{axis} [{i1}, {i2}])'
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, label=f'Number of counts ({method})')
        plt.tight_layout()
        if png is not None:
            plt.savefig(png, metadata={'Software': 'extract_2d_slice_from_3d_cube'})
            plt.close()
        else:
            plt.show()

    # save result
    if output is not None:
        hdu = fits.PrimaryHDU(slice2d.astype(np.float32))
        header_output = hdu.header

        if wavecal != "none":
            if wavecal in "123":
                axiswave = int(wavecal)
                header_output['ctype1'] = header[f'ctype{axiswave}']
                header_output['crpix1'] = header[f'crpix{axiswave}']
                header_output['crval1'] = header[f'crval{axiswave}']
                header_output['cdelt1'] = header[f'cdelt{axiswave}']
            elif wavecal == "fridasimulator":
                axiswave = 3
                header_output['ctype1'] = header[f'ctype{axiswave}']
                header_output['crpix1'] = header[f'crpix{axiswave}']
                header_output['crval1'] = header[f'crval{axiswave}']
                header_output['cdelt1'] = header[f'pc{axiswave}_{axiswave}']
            else:
                raise ValueError(f"Unexpected {wavecal=}")

        header_output['history'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        header_output['comment'] = 'Image created with extract_2d_slice_from_3d_cube.py'
        header_output['comment'] = 'with the following arguments:'
        header_output['comment'] = f'input: {input}'
        header_output['comment'] = f'axis: {axis}'
        header_output['comment'] = f'i1: {i1}'
        header_output['comment'] = f'i2: {i2}'
        header_output['comment'] = f'--method: {method}'
        header_output['comment'] = f'--wavecal: {wavecal}'
        header_output['comment'] = f'--transpose: {transpose}'
        header_output['comment'] = f'output: {output}'

        hdu.writeto(output, overwrite="yes")


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(description="Extract 2D slice from 3D cube", formatter_class=RichHelpFormatter)
    parser.add_argument("input", help="Input FITS file")
    parser.add_argument("--axis", help="Axis to be collapsed in output", type=int, default=3)
    parser.add_argument("--i1", help="First pixel of the projected axis", type=int, default=1)
    parser.add_argument("--i2", help="Last pixel of the projected axis (0=NAXIS value)", type=int, default=0)
    parser.add_argument("--method", help="Collapse method (default=sum)", type=str, default="sum",
                        choices=["sum", "mean", "median"])
    parser.add_argument("--wavecal", help="Wavelength calibration type", type=str, default="none",
                        choices=["none", "fridasimulator", "1", "2", "3"])
    parser.add_argument("--transpose", help="Transpose data array in output", action="store_true")
    parser.add_argument("--noplot", help="Do not plot result", action="store_true")
    parser.add_argument("--vmin", help="vmin value for imshow", type=float)
    parser.add_argument("--vmax", help="vmax value for imshow", type=float)
    parser.add_argument("--output", help="Output FITS file")
    parser.add_argument("--png", help="Output PNG file (plot of the result)", type=str)
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    parser.add_argument("--debug", help="Debug", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.debug:
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    extract_slice(
        input=args.input,
        axis=args.axis,
        i1=args.i1,
        i2=args.i2,
        method=args.method,
        wavecal=args.wavecal,
        transpose=args.transpose,
        vmin=args.vmin,
        vmax=args.vmax,
        noplot=args.noplot,
        output=args.output,
        png=args.png
    )


if __name__ == "__main__":

    main()
