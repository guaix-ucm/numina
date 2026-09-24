#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Show WCS info of a particular FITS image extension."""

import argparse
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime
import logging
from pathlib import Path
from rich_argparse import RichHelpFormatter
import sys

from numina.tools.initialize_script_with_args import include_default_arguments_for_common_actions
from numina.tools.initialize_script_with_args import initialize_script_with_args
from numina.tools.initialize_script_with_args import goodbye_message_and_save_console

from numina._version import __version__

from .file_is_valid_fits import file_is_valid_fits


def show_wcs_info(list_of_fits_files, extname_image):
    """Show WCS info of a particular FITS image extension.
    
    Parameters
    ----------
    list_of_fits_files : list
        List of FITS files to process.
    extname_image : str
        Extension name for image in input files. Default value: PRIMARY.
    """
    logger = logging.getLogger(__name__)

    # protections
    if not isinstance(list_of_fits_files, list):
        raise TypeError("list_of_fits_files must be a list")

    nimages = len(list_of_fits_files)
    for i, fname in enumerate(list_of_fits_files):
        with fits.open(fname) as hdul:
            if extname_image not in hdul:
                raise ValueError(f"Expected {extname_image} extension not found")
            hdu = hdul[extname_image]
            logger.info(f"\n--- Image {i+1}/{nimages} ---\n")
            logger.info(f"Working with file (extension): {fname} ({extname_image})")
            for i in range(1, 4):
                key = f"NAXIS{i}"
                if key in hdu.header:
                    logger.info(f"{key} = {hdu.header[key]}")
            wcs = WCS(hdu.header)
            logger.info(f"WCS info:\n{wcs.to_header_string()}")


def main(args=None):

    datetime_ini = datetime.now()

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="Show WCS info of a particual FITS image extension", formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "input_list", help="TXT file with list of 3D images to be combined or single FITS file", type=str, nargs="+"
    )
    parser.add_argument(
        "--extname-image",
        help="Extension name for image in input files. Default value: PRIMARY",
        default="PRIMARY",
        type=str,
    )
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger = initialize_script_with_args(sys.argv, parser, args, __name__, __version__)

    input_list = args.input_list
    extname_image = args.extname_image

    # If input is a single FITS file, use it directly; 
    # otherwise, read the list of files from the provided file
    if len(input_list) == 1:
        if input_list[0].lower().endswith(".fits"):
            file_content = [input_list[0]]
        else:
            with open(input_list[0]) as f:
                file_content = f.read().splitlines()
    else:
        file_content = input_list

    list_of_fits_files = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ["#"]:
                if not Path(fname).is_file():
                    raise ValueError(f"File {fname} does not exist or is not a valid file.")
                if not file_is_valid_fits(fname):
                    raise ValueError(f"File {fname} is not a valid FITS file.")
                list_of_fits_files.append(fname)

    if len(list_of_fits_files) < 1:
        raise ValueError(f"No valid FITS files found in {input_list}. Please check the file content.")

    # Show WCS info
    show_wcs_info(list_of_fits_files, extname_image)

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":
    main()
