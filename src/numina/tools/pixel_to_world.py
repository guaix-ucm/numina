#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute pixel_to_world.
"""
import argparse

from astropy.io import fits
from astropy.wcs import WCS
from rich import print
from rich_argparse import RichHelpFormatter
import sys


def pixel_to_world(inputfile, pixel, extnum, verbose=False):
    """Compute world_to_pixel.

    Parameters
    ----------
    inputfile : str, file-like or `pathlib.Path`
        Input FITS filename.
    pixel : str or None
        WCS pixel coordinate.
    extnum : int
        Extension number to read the WCS from.
    verbose : bool
        It True, display intermediate information
    """

    with fits.open(inputfile) as hdul:
        if verbose:
            print(hdul.info())
        if extnum > len(hdul):
            raise ValueError(f"Extension number {extnum} exceeds {len(hdul)}")
        header = hdul[extnum].header

    wcs = WCS(header)
    if verbose:
        print(wcs)
    naxis = wcs.naxis

    if pixel is None or pixel == "":
        pixel = [1] * naxis
    else:
        pixel = [float(item) - 1 for item in pixel.split(',')]   # Python criterion for next function

    result = wcs.pixel_to_world(*pixel)

    print(f'\nComputing world coordinates for pixel {pixel}:')

    if isinstance(result, list):
        print(result[0])
        print(result[1])
    else:
        print(result)


def main(args=None):
    """
    Usage example:
    $ numina-pixel_to_world file.fits --pixel '1,1'

    The pixel coordinates are read as a string. The quote or double
    quote symbol is not necessary if the numbers are given without
    blank spaces.
    """
    parser = argparse.ArgumentParser(description="Convert pixel to world coordinates.",
                                     formatter_class=RichHelpFormatter)
    parser.add_argument("inputfile", help="Input FITS file", type=str)
    parser.add_argument("--pixel", help="WCS pixel coordinate (comma separated values)", type=str,
                        default=None)
    parser.add_argument("--extnum", help="Extension number (default 0=PRIMARY)", type=int, default=0)
    parser.add_argument("--verbose", help="Display intermediate information", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

    if args.echo:
        print('[bold red]Executing:\n' + ' '.join(sys.argv) + '[/bold red]')

    extnum = args.extnum
    if extnum < 0:
        raise ValueError('extnum must be >= 0')

    pixel_to_world(
        inputfile=args.inputfile,
        pixel=args.pixel,
        extnum=extnum,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
