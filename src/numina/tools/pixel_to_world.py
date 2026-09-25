#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute pixel_to_world."""

import argparse
from astropy.io import fits
from astropy.coordinates import SkyCoord, SpectralCoord
import io
import logging
import numpy as np
from rich_argparse import RichHelpFormatter
import sys

from .hdul_utils import get_hdu_from_hdul, get_wcs_from_hdu
from .initialize_script_with_args import include_default_arguments_for_common_actions
from .initialize_script_with_args import initialize_script_with_args
from .initialize_script_with_args import goodbye_message_and_save_console


def pixel_to_world(inputfile, pixel, extnum, extname, wcskey):
    """Compute world_to_pixel.

    Parameters
    ----------
    inputfile : str, file-like or `pathlib.Path`
        Input FITS filename.
    pixel : str or None
        WCS pixel coordinate.
    extnum : int
        Extension number to read the WCS from.
    extname : str
        Extension name to read the WCS from.
    wcskey : str
        WCS key to use when multiple WCS are present in the FITS header.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Opened FITS file: {inputfile}")

    with fits.open(inputfile) as hdul:
        buffer = io.StringIO()
        hdul.info(output=buffer)
        logger.info(buffer.getvalue().rstrip(), extra={"markup": False})
        hdu = get_hdu_from_hdul(hdul, extnum=extnum, extname=extname)
        wcs = get_wcs_from_hdu(hdu, wcskey=wcskey)
        header = hdu.header

    logger.debug(f"WCS info: {wcs}")
    naxis = wcs.naxis

    for i in range(1, naxis + 1):
        ctype = header.get(f"CTYPE{i}", "")
        cunit = header.get(f"CUNIT{i}", "")
        if ctype and not (isinstance(cunit, str) and cunit.strip()):
            logger.error(f"CUNIT{i} is missing or undefined for CTYPE{i}='{ctype}'")
            sys.exit(1)

    if pixel is None or pixel == "":
        pixel = np.ones(naxis, dtype=float).tolist()
    elif isinstance(pixel, str):
        pixel = [float(item) for item in pixel.split(",")]
    logger.debug(f"Pixel coordinates: {pixel}")
    pixel_python = [float(item) - 1 for item in pixel]  # Python criterion for next function

    result = wcs.pixel_to_world(*pixel_python)

    logger.info(f"\nComputing world coordinates for pixel {pixel}:")

    if isinstance(result, list):
        for item in result:
            if isinstance(item, SkyCoord):
                logger.info(f"SkyCoord: {item.to_string()}")
            elif isinstance(item, SpectralCoord):
                logger.info(f"SpectralCoord: {item.to_string()}")
            else:
                logger.info(f"{item.to_string()}")
    else:
        logger.info(result.to_string())


def main(args=None):
    """
    Usage example:
    $ numina-pixel_to_world file.fits --pixel '1,1'

    The pixel coordinates are read as a string. The quote or double
    quote symbol is not necessary if the numbers are given without
    blank spaces.
    """
    parser = argparse.ArgumentParser(
        description="Convert pixel to world coordinates.", formatter_class=RichHelpFormatter
    )
    parser.add_argument("inputfile", help="Input FITS file", type=str)
    parser.add_argument("--pixel", help="WCS pixel coordinate (comma separated values)", type=str, default=None)
    parser.add_argument("-e", "--extnum", help="Extension number", type=int)
    parser.add_argument("--extname", help="Extension name", type=str)
    parser.add_argument(
        "--wcskey", help="WCS key to use when multiple WCS are present in the FITS header", type=str, default=None
    )
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger, datetime_ini = initialize_script_with_args(sys.argv, parser, args, __name__)

    pixel_to_world(
        inputfile=args.inputfile,
        pixel=args.pixel,
        extnum=args.extnum,
        extname=args.extname,
        wcskey=args.wcskey,
    )

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":
    main()
