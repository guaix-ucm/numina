#
# Copyright 2019-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Auxiliary script to perform binary image arithmetic."""

import argparse
import sys

from astropy.io import fits
import numpy as np

from numina.array.display.ximshow import ximshow_file


def compute_operation(file1, file2, operation, output,
                      overwrite=True,
                      display='none',
                      args_z1z2=None,
                      args_bbox=None,
                      args_keystitle=None,
                      args_geometry=None):
    """Compute output = file1 operation file2.

    Parameters
    ----------
    file1 : file object
        First FITS file.
    file2 : file object or float
        Second FITS file or float number.
    operation : string
        Mathematical operation.
    output : file object
        Output FITS file.
    overwrite : bool
        If True, the output file can be overwritten.
    display : string
        Character string indication whether the images are displayed.
        Valid values are 'all', 'result' and 'none' (default).
    args_z1z2 : string or None
        String providing the image cuts tuple: z1, z2, minmax or None.
    args_bbox : string or None
        String providing the bounding box tuple: nc1, nc2, ns1, ns2.
    args_keystitle : string or None
        Tuple of FITS keywords.format: key1,key2,...,keyn.format
    args_geometry : string or None
        Tuple x,y,dx,dy to define the Qt backend geometry.

    """

    # read first FITS file
    with fits.open(file1) as hdulist:
        image_header1 = hdulist[0].header
        image1 = hdulist[0].data.astype(float)
    naxis = image_header1['naxis']
    naxis1 = image_header1['naxis1']
    naxis2 = image_header1['naxis2']

    # if required, display file1
    if display == 'all':
        ximshow_file(file1.name,
                     args_z1z2=args_z1z2, args_bbox=args_bbox,
                     args_keystitle=args_keystitle,
                     args_geometry=args_geometry,
                     debugplot=12)

    # read second FITS file or number
    try:
        with fits.open(file2) as hdulist:
            image_header2 = hdulist[0].header
            image2 = hdulist[0].data.astype(float)
            naxis_ = image_header2['naxis']
            naxis1_ = image_header2['naxis1']
            naxis2_ = image_header2['naxis2']
            filename = file2
    except FileNotFoundError:
        image2 = np.zeros((naxis2, naxis1), dtype=float)
        image2 += float(file2)
        naxis_ = naxis
        naxis1_ = naxis1
        naxis2_ = naxis2
        filename = 'constant=' + str(file2)

    # if required, display file2
    if display == 'all':
        ximshow_file(filename,
                     args_z1z2=args_z1z2, args_bbox=args_bbox,
                     args_keystitle=args_keystitle,
                     args_geometry=args_geometry,
                     debugplot=12)

    # second image is a single image row
    if naxis1 == naxis1_ and naxis2 > naxis2_:
        if naxis2_ == 1:
            image2 = np.tile(image2, naxis2).reshape((naxis2, naxis1))
            naxis2_ = naxis2

    # second image is a single image column
    if naxis2 == naxis2_ and naxis1 > naxis1_:
        if naxis1_ == 1:
            image2 = np.repeat(image2, naxis1).reshape((naxis2, naxis1))
            naxis1_ = naxis1

    # additional dimension checks
    if naxis != naxis_:
        raise ValueError("NAXIS1 values are different.")
    if naxis1 != naxis1_:
        raise ValueError("NAXIS1 values are different.")
    if naxis2 != naxis2_:
        raise ValueError("NAXIS2 values are different.")

    # compute operation
    if operation == "+":
        solution = image1 + image2
    elif operation == "-":
        solution = image1 - image2
    elif operation == "x":
        solution = image1 * image2
    elif operation == "/":
        solution = image1 / image2
    else:
        raise ValueError("Unexpected operation=" + str(operation))

    # save output file
    hdu = fits.PrimaryHDU(solution.astype(float), image_header1)
    hdu.writeto(output, overwrite=overwrite)

    # if required, display result
    if display in ['all', 'result']:
        ximshow_file(output,
                     args_z1z2=args_z1z2, args_bbox=args_bbox,
                     args_keystitle=args_keystitle,
                     args_geometry=args_geometry,
                     debugplot=12)


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
                        choices=['+', '-', 'x', '/'])
    parser.add_argument("file2",
                        help="Second FITS image or number",
                        type=str)
    parser.add_argument("output",
                        help="Output FITS image",
                        type=str)
    # optional arguments
    parser.add_argument("--overwrite",
                        help="Overwrite output file if already exists",
                        action="store_true")
    parser.add_argument("--display",
                        help="Display images: all, result, none (default)",
                        default="none",
                        type=str,
                        choices=['all', 'result', 'none'])
    parser.add_argument("--z1z2",
                        help="tuple z1,z2, minmax or None (use zscale)")
    parser.add_argument("--bbox",
                        help="bounding box tuple: nc1,nc2,ns1,ns2")
    parser.add_argument("--keystitle",
                        help="tuple of FITS keywords.format: " +
                             "key1,key2,...keyn.'format'")
    parser.add_argument("--geometry",
                        help="Tuple x,y,dx,dy indicating window geometry",
                        default="0,0,640,480")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # compute operation
    compute_operation(args.file1, args.file2,
                      args.operation, args.output, args.overwrite,
                      args.display,
                      args.z1z2, args.bbox, args.keystitle, args.geometry)


if __name__ == "__main__":

    main()
