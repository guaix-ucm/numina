#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Auxiliary script to perform binary image arithmetic.

The computation is performed casting to np.float32.
"""

import argparse
import sys

from astropy.io import fits
import numpy as np
import pathlib

from numina.array.display.ximshow import ximshow_file, ximshow


def extnum_from_extname(file, extname):
    """Get the extension number from the extension name in a FITS file.

    Parameters
    ----------
    file : file object
        FITS file.
    extname : str
        Extension name.

    Returns
    -------
    int
        Extension number.
    """
    with fits.open(file) as hdulist:
        for i, hdu in enumerate(hdulist):
            if hdu.name.lower() == extname.lower():
                return i

    raise ValueError(f"Extension '{extname}' not found in {file.name}.")


def compute_operation(
    file1,
    file2,
    operation,
    extname1,
    extname2,
    output,
    overwrite=True,
    dtype='float32',
    display='none',
    args_z1z2=None,
    args_bbox=None,
    args_keystitle=None,
    args_geometry=None
):
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
        Extension name of the first FITS file.
    extname2 : str
        Extension name of the second FITS file.
    output : file object
        Output FITS file.
    overwrite : bool
        If True, the output file can be overwritten.
    dtype : str
        Data type of the output image. Default is 'float32'.
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
        image_header1 = hdulist[extname1].header
        image1 = hdulist[extname1].data.astype(dtype)
    naxis = image_header1['naxis']
    naxis1 = image_header1['naxis1']
    naxis2 = image_header1['naxis2']

    # if required, display file1
    if display == 'all':
        ximshow_file(file1,
                     extnum=extnum_from_extname(file1, extname1)+1,  # extnum is 1-based in ximshow_file
                     args_z1z2=args_z1z2, args_bbox=args_bbox,
                     args_keystitle=args_keystitle,
                     args_geometry=args_geometry,
                     debugplot=12)

    # read second FITS file or number
    file2_is_a_file = True
    try:
        with fits.open(file2) as hdulist:
            image_header2 = hdulist[extname2].header
            image2 = hdulist[extname2].data.astype(dtype)
            naxis_ = image_header2['naxis']
            naxis1_ = image_header2['naxis1']
            naxis2_ = image_header2['naxis2']
            filename = file2
    except FileNotFoundError:
        file2_is_a_file = False
        image2 = np.zeros((naxis2, naxis1), dtype=dtype)
        if 'int' in dtype:
            image2 += int(file2)
        elif 'float' in dtype:
            image2 += float(file2)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use an integer or float type.")
        naxis_ = naxis
        naxis1_ = naxis1
        naxis2_ = naxis2
        filename = 'constant=' + str(file2)

    # if required, display file2
    if display == 'all':
        if file2_is_a_file:
            ximshow_file(file2,
                         extnum=extnum_from_extname(file2, extname2)+1,  # extnum is 1-based in ximshow_file
                         args_z1z2=args_z1z2, args_bbox=args_bbox,
                         args_keystitle=args_keystitle,
                         args_geometry=args_geometry,
                         debugplot=12)
        else:
            ximshow(image2,
                    title=filename,
                    z1z2=args_z1z2, image_bbox=args_bbox,
                    geometry=args_geometry,
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
        raise ValueError("NAXIS values are different.")
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
    elif operation == "=":
        solution = image2.copy()
    else:
        raise ValueError("Unexpected operation=" + str(operation))

    # save output file
    hdu = fits.PrimaryHDU(solution.astype(dtype), image_header1)
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

    if not args.overwrite and pathlib.Path(args.output).exists():
        print(f"Output file {args.output} already exists. Use --overwrite to overwrite it.")
        sys.exit(1)

    # compute operation
    compute_operation(
        file1=args.file1,
        file2=args.file2,
        operation=args.operation,
        extname1=args.extname1,
        extname2=args.extname2,
        output=args.output,
        overwrite=args.overwrite,
        dtype=args.dtype,
        display=args.display,
        args_z1z2=args.z1z2,
        args_bbox=args.bbox,
        args_keystitle=args.keystitle,
        args_geometry=args.geometry
    )


if __name__ == "__main__":

    main()
