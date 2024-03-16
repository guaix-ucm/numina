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
import numpy as np
import sys


def extract_slice(input, axis3, i1, i2, method, wavecal, transpose, output):
    """Extract slice.

    Parameters
    ----------
    input : str
        Input FITS file name.
    axis3 : int
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
    output : str
        Output FITS file name.

    """

    # protections
    if not (1 <= axis3 <= 3):
        raise ValueError(f'{axis3=} out of valid range (1, 2 or 3)')

    # read first FITS file
    with fits.open(input) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data.astype(float)
    naxis = header['naxis']
    if naxis != 3:
        raise ValueError(f'Unexpected input {naxis=} (it must be 3)')
    naxis3 = header[f'naxis{axis3}']

    # set last pixel to maximum value when i2==0
    if i2 == 0:
        i2 = naxis3

    if not (1 <= i1 <= naxis3):
        raise ValueError(f'Unexpected {i1=}. Must be a number in [1,...,{naxis3}]')
    if not (1 <= i2 <= naxis3):
        raise ValueError(f'Unexpected {i2=}. Must be a number in [1,...,{naxis3}]')
    if i2 < i1:
        raise ValueError(f'{i1=} must be <= {i2=}')

    if axis3 == 1:
        if method == "sum":
            slice = np.sum(data[:, :, (i1-1):i2], axis=2, keepdims=False)
        elif method == "mean":
            slice = np.mean(data[:, :, (i1 - 1):i2], axis=2, keepdims=False)
        elif method == "median":
            slice = np.median(data[:, :, (i1 - 1):i2], axis=2, keepdims=False)
        else:
            raise ValueError(f'Unexpected {method=}')
    elif axis3 == 2:
        if method == "sum":
            slice = np.sum(data[:, (i1-1):i2, :], axis=1, keepdims=False)
        elif method == "mean":
            slice = np.mean(data[:, (i1 - 1):i2, :], axis=1, keepdims=False)
        elif method == "median":
            slice = np.median(data[:, (i1 - 1):i2, :], axis=1, keepdims=False)
        else:
            raise ValueError(f'Unexpected {method=}')
    else:
        if method == "sum":
            slice = np.sum(data[(i1-1):i2, :, :], axis=0, keepdims=False)
        elif method == "mean":
            slice = np.mean(data[(i1-1):i2, :, :], axis=0, keepdims=False)
        elif method == "median":
            slice = np.median(data[(i1-1):i2, :, :], axis=0, keepdims=False)
        else:
            raise ValueError(f'Unexpected {method=}')

    if transpose:
        slice = np.transpose(slice)

    # save result
    hdu = fits.PrimaryHDU(slice.astype(np.float32))
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
    header_output['comment'] = 'Image created with r6_extract_2d_slice_from_3d_cube.py'
    header_output['comment'] = 'with the following arguments:'
    header_output['comment'] = f'input: {input}'
    header_output['comment'] = f'axis3: {axis3}'
    header_output['comment'] = f'i1: {i1}'
    header_output['comment'] = f'i2: {i2}'
    header_output['comment'] = f'--method: {method}'
    header_output['comment'] = f'--wavecal: {wavecal}'
    header_output['comment'] = f'--transpose: {transpose}'
    header_output['comment'] = f'output: {output}'


    hdu.writeto(output, overwrite="yes")


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input FITS file")
    parser.add_argument("axis3", help="Axis to be collapsed in output", type=int)
    parser.add_argument("i1", help="First pixel of the projected axis", type=int)
    parser.add_argument("i2", help="Last pixel of the projected axis (0=NAXIS value)", type=int)
    parser.add_argument("output", help="Output FITS file")
    parser.add_argument("--method", help="Collapse method (default=sum)", type=str, default="sum",
                        choices=["sum", "mean", "median"])
    parser.add_argument("--wavecal", help="Wavelength calibration type", type=str, default="none",
                        choices=["none", "fridasimulator", "1", "2", "3"])
    parser.add_argument("--transpose", help="Transpose data array in output", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    extract_slice(
        input=args.input,
        axis3=args.axis3,
        i1=args.i1,
        i2=args.i2,
        method=args.method,
        wavecal=args.wavecal,
        transpose=args.transpose,
        output=args.output
    )


if __name__ == "__main__":

    main()
