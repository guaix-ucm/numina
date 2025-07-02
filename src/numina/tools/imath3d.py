#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Auxiliary script to perform binary image arithmetic with 3D cubes.

The computation is performed casting to np.float32."""

import argparse
import sys

from astropy.io import fits
import numpy as np
import pathlib

from .add_script_info_to_fits_history import add_script_info_to_fits_history

def compute_operation(file1, file2, operation, extname1, extname2, dtype):
    """Compute output = file1 operation file2.

    Parameters
    ----------
    file1 : file object
        First FITS file.
    file2 : file object or float
        Second FITS file or float number.
    operation : string
        Mathematical operation.
    extname1 : str
        Extension name of the first FITS file (default: 'PRIMARY').
    extname2 : str
        Extension name of the second FITS file (default: 'PRIMARY').
    dtype : str
        Data type of the output image.

    Returns
    -------
    solution : `numpy.ndarray`
        Resulting 3D image after applying the operation.
    """

    # read first FITS file
    with fits.open(file1) as hdulist:
        image_header1 = hdulist[extname1].header
        image1 = hdulist[extname1].data.astype(dtype)
    naxis = image_header1['naxis']
    if naxis != 3:
        raise ValueError("Input file must be a 3D cube (NAXIS=3).")
    naxis1 = image_header1['naxis1']
    naxis2 = image_header1['naxis2']
    naxis3 = image_header1['naxis3']

    # read second FITS file or number
    try:
        with fits.open(file2) as hdulist:
            image_header2 = hdulist[extname2].header
            image2 = hdulist[extname2].data.astype(dtype)
            naxis_ = image_header2['naxis']
            naxis1_ = image_header2['naxis1']
            naxis2_ = image_header2['naxis2']
            naxis3_ = image_header2['naxis3']
    except FileNotFoundError:
        image2 = np.zeros((naxis3, naxis2, naxis1), dtype=dtype)
        if 'int' in dtype:
            file2 = int(file2)
        elif 'float' in dtype:
            file2 = float(file2)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use an integer or float type.")
        image2 += file2  # if file2 is a number
        naxis_ = naxis
        naxis1_ = naxis1
        naxis2_ = naxis2
        naxis3_ = naxis3

    # additional dimension checks
    if naxis != naxis_:
        raise ValueError("NAXIS values are different.")
    if naxis1 != naxis1_:
        raise ValueError("NAXIS1 values are different.")
    if naxis2 != naxis2_:
        raise ValueError("NAXIS2 values are different.")
    if naxis3 != naxis3_:
        raise ValueError("NAXIS3 values are different.")

    # compute operation
    if operation == "+":
        solution = image1 + image2
    elif operation == "-":
        solution = image1 - image2
    elif operation == "x":
        solution = image1 * image2
    elif operation == "/":
        solution = image1 / image2
    elif operation == "=":
        solution = image2.copy()
    else:
        raise ValueError("Unexpected operation=" + str(operation))

    return solution


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="description: binary image arithmetic"
    )
    # positional parameters
    parser.add_argument("file1",
                        help="First FITS image",
                        type=argparse.FileType('rb'))
    parser.add_argument("operation",
                        help="Arithmetic operation",
                        type=str,
                        choices=['+', '-', 'x', '/', '='])
    parser.add_argument("file2",
                        help="Second FITS image or number",
                        type=str)
    parser.add_argument("output",
                        help="Output FITS image",
                        type=str)
    # optional arguments
    parser.add_argument("--extname1", type=str,
                        help="Extension name of the first FITS file (default: 'PRIMARY').",
                        default='PRIMARY')
    parser.add_argument("--extname2", type=str,
                        help="Extension name of the second FITS file (default: 'PRIMARY').",
                        default='PRIMARY')
    parser.add_argument("--overwrite",
                        help="Overwrite output file if already exists",
                        action="store_true")
    parser.add_argument("--dtype", 
                        help="Data type of the output image (default: float32)",
                        type=str, 
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64',
                                 'float32', 'float64'],
                        default='float32')
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    if not args.overwrite and pathlib.Path(args.output).exists():
        print(f"Output file {args.output} already exists. Use --overwrite to overwrite it.")
        sys.exit(1)

    # compute operation
    solution = compute_operation(
        file1=args.file1, 
        file2=args.file2,
        operation=args.operation, 
        extname1=args.extname1,
        extname2=args.extname2,
        dtype=args.dtype
    )

    # save output file
    hdu = fits.PrimaryHDU(solution.astype(args.dtype))
    add_script_info_to_fits_history(hdu.header, args)
    hdu.writeto(args.output, overwrite=args.overwrite)


if __name__ == "__main__":

    main()
