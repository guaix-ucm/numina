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
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SpectralCoord
import logging
import numpy as np
from rich_argparse import RichHelpFormatter
import sys

from numina.tools.initialize_script_with_args import include_default_arguments_for_common_actions
from numina.tools.initialize_script_with_args import initialize_script_with_args
from numina.tools.initialize_script_with_args import goodbye_message_and_save_console

from numina._version import __version__


def pixel_to_world(inputfile, pixel, extnum):
    """Compute world_to_pixel.

    Parameters
    ----------
    inputfile : str, file-like or `pathlib.Path`
        Input FITS filename.
    pixel : str or None
        WCS pixel coordinate.
    extnum : int
        Extension number to read the WCS from.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Opened FITS file: {inputfile}")

    with fits.open(inputfile) as hdul:
        logger.debug(hdul.info())
        if extnum > len(hdul) - 1:
            raise ValueError(f"Extension number {extnum} exceeds {len(hdul) - 1}")
        header = hdul[extnum].header

    wcs = WCS(header)
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
    parser.add_argument("-e", "--extnum", help="Extension number (default 0=PRIMARY)", type=int, default=0)
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger, datetime_ini = initialize_script_with_args(sys.argv, parser, args, __name__, __version__)

    extnum = args.extnum
    if extnum < 0:
        raise ValueError("extnum must be >= 0")

    pixel_to_world(
        inputfile=args.inputfile,
        pixel=args.pixel,
        extnum=extnum,
    )

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":
    main()
