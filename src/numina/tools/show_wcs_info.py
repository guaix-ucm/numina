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
import io
import logging
from pathlib import Path
from rich_argparse import RichHelpFormatter
import sys

from .file_is_valid_fits import file_is_valid_fits
from .hdul_utils import get_wcs_from_hdul
from .initialize_script_with_args import include_default_arguments_for_common_actions
from .initialize_script_with_args import initialize_script_with_args
from .initialize_script_with_args import goodbye_message_and_save_console


def show_wcs_info(list_of_fits_files, extname=None, extnum=None, wcskey=None):
    """Show WCS info of a particular FITS image extension.

    Parameters
    ----------
    list_of_fits_files : list
        List of FITS files to process.
    extname : str
        Extension name for image in input files.
    extnum : int
        Extension number for image in input files.
    wcskey : str
    """
    logger = logging.getLogger(__name__)

    # protections
    if not isinstance(list_of_fits_files, list):
        raise TypeError("list_of_fits_files must be a list")

    nimages = len(list_of_fits_files)
    for i, fname in enumerate(list_of_fits_files):
        logger.info(f"\n--- Image {i+1}/{nimages} ---\n")
        with fits.open(fname) as hdul:
            buffer = io.StringIO()
            hdul.info(output=buffer)
            logger.info(buffer.getvalue().rstrip(), extra={"markup": False})
            wcs = get_wcs_from_hdul(hdul, extname=extname, extnum=extnum, wcskey=wcskey)
            naxis = wcs.naxis
            logger.info(f"NAXIS = {naxis} {wcs.pixel_shape}")
            logger.info(f"CTYPE = {list(wcs.wcs.ctype)}")
            header_text = wcs.to_header().tostring(sep="\n", endcard=False, padding=False).rstrip()
            logger.info(f"WCS info:\n{header_text}", extra={"markup": False})


def main(args=None):
    """Main function to show WCS info of a particular FITS image extension."""

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="Show WCS info of a particual FITS image extension", formatter_class=RichHelpFormatter
    )
    parser.add_argument(
        "input_list", help="TXT file with list of 3D images to be combined or single FITS file", type=str, nargs="+"
    )
    parser.add_argument(
        "-e",
        "--extnum",
        help="Extension number for image in input files.",
        type=int,
    )
    parser.add_argument(
        "--extname",
        help="Extension name for image in input files.",
        type=str,
    )
    parser.add_argument(
        "--wcskey",
        help="WCS key to use when multiple WCS are present in the FITS header.",
        type=str,
    )
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger, datetime_ini = initialize_script_with_args(sys.argv, parser, args, __name__)

    input_list = args.input_list
    extnum = args.extnum
    extname = args.extname
    wcskey = args.wcskey
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
    show_wcs_info(list_of_fits_files, extname, extnum, wcskey)

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":
    main()
