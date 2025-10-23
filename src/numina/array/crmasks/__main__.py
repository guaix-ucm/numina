#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Combination of arrays avoiding coincident cosmic ray hits."""
import logging
import os
import sys

import argparse
from astropy.io import fits
import numpy as np
from rich.logging import RichHandler
import yaml

from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.user.console import NuminaConsole
from numina._version import __version__

from .__main__ import VALID_COMBINATIONS
from .apply_crmasks import apply_crmasks
from .compute_crmasks import compute_crmasks


def main(args=None):
    """
    Main function to compute and apply CR masks.
    """

    parser = argparse.ArgumentParser(
        description="Combine 2D arrays using mediancr, meancrt or meancr methods."
    )

    parser.add_argument("inputyaml",
                        help="Input YAML file.",
                        type=str)
    parser.add_argument("--crmasks",
                        help="FITS file with cosmic ray masks",
                        type=str)
    parser.add_argument("--log-level",
                        help="Set the logging level",
                        type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO')
    parser.add_argument("--record",
                        help="Record terminal output",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    # Configure rich console
    console = NuminaConsole(record=args.record)

    if args.echo:
        console.print(f"[bright_red]Executing: {' '.join(sys.argv)}[/bright_red]\n", end='')

    # Configure logging
    if args.log_level in ['DEBUG', 'WARNING', 'ERROR', 'CRITICAL']:
        format_log = '%(name)s %(levelname)s %(message)s'
        handlers = [RichHandler(console=console, show_time=False, markup=True)]
    else:
        format_log = '%(message)s'
        handlers = [RichHandler(console=console, show_time=False, markup=True, show_path=False, show_level=False)]
    logging.basicConfig(
        level=args.log_level,
        format=format_log,
        handlers=handlers
    )
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Suppress matplotlib debug logs

    # Display version info
    logger = logging.getLogger(__name__)
    logger.info(f"Using {__name__} version {__version__}")

    # Read parameters from YAML file
    with open(args.inputyaml, 'rt') as fstream:
        input_params = yaml.safe_load(fstream)
    logger.debug(f'{input_params=}')

    # Check that mandatory parameters are present
    if 'images' not in input_params:
        raise ValueError("'images' must be provided in input YAML file.")
    else:
        list_of_fits_files = input_params['images']
        if not isinstance(list_of_fits_files, list) or len(list_of_fits_files) < 3:
            raise ValueError("'images' must be a list of at least 3 FITS files.")
        for file in list_of_fits_files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File {file} not found.")
            else:
                logger.info("found input file: %s", file)
    for item in ['gain', 'rnoise', 'bias']:
        if item not in input_params:
            raise ValueError(f"'{item}' must be provided in input YAML file.")

    # Default values for missing parameters in input YAML file
    if 'extnum' in input_params:
        extnum = int(input_params['extnum'])
    else:
        extnum = 0
        logger.info("extnum not provided, assuming extnum=0")

    # Read the input list of files, which should contain paths to 2D FITS files,
    # and load the arrays from the specified extension number.
    list_arrays = [fits.getdata(file, extnum=extnum) for file in input_params['images']]

    # Check if the list is empty
    if not list_arrays:
        raise ValueError("The input list is empty. Please provide a valid list of 2D arrays.")

    # Check that the requirements are provided
    if 'requirements' not in input_params:
        raise ValueError("'requirements' must be provided in input YAML file.")
    requirements = input_params['requirements']
    if not isinstance(requirements, dict):
        raise ValueError("'requirements' must be a dictionary.")
    if not requirements:
        raise ValueError("'requirements' dictionary is empty.")

    # Define parameters for compute_crmasks
    crmasks_params = dict()
    for key in ['gain', 'rnoise', 'bias']:
        crmasks_params[key] = input_params[key]
    for item in input_params['requirements']:
        crmasks_params[item] = input_params['requirements'][item]

    # If a FITS file with cosmic ray masks is provided, read it and skip
    # the computation of the masks. Otherwise, compute the masks.
    if args.crmasks is None:
        # Compute the different cosmic ray masks
        console.rule("[bold magenta] Computing cosmic ray masks [/bold magenta]")
        hdul_masks = compute_crmasks(
            list_arrays=list_arrays,
            _logger=logger,
            debug=(args.log_level == 'DEBUG'),
            **crmasks_params
        )
        # Save the cosmic ray masks to a FITS file
        output_masks = 'crmasks.fits'
        logger.info("Saving cosmic ray masks to %s", output_masks)
        hdul_masks.writeto(output_masks, overwrite=True)
        logger.info("Cosmic ray masks saved")
    else:
        if not os.path.isfile(args.crmasks):
            raise FileNotFoundError(f"File {args.crmasks} not found.")
        else:
            logger.info("reading cosmic ray masks from %s", args.crmasks)
            hdul_masks = fits.open(args.crmasks)

    # Apply cosmic ray masks
    for combination in VALID_COMBINATIONS:
        console.rule(f"Applying cosmic ray masks using [bold magenta]{combination}[/bold magenta]")
        output_combined = f'combined_{combination}.fits'
        combined, variance, maparray = apply_crmasks(
            list_arrays=list_arrays,
            bias=input_params['bias'],
            hdul_masks=hdul_masks,
            combination=combination,
            use_lamedian=crmasks_params.get('use_lamedian', False),
            dtype=np.float32
        )
        # Save the combined array, variance, and map to a FITS file
        logger.info("Saving combined (bias subtracted) array, variance, and map to %s", output_combined)
        hdu_combined = fits.PrimaryHDU(combined.astype(np.float32))
        add_script_info_to_fits_history(hdu_combined.header, args)
        hdu_combined.header.add_history('Contents of --inputlist:')
        for item in list_of_fits_files:
            hdu_combined.header.add_history(f'- {item}')
        hdu_combined.header.add_history(f"Masks UUID: {hdul_masks[0].header['UUID']}")
        hdu_variance = fits.ImageHDU(variance.astype(np.float32), name='VARIANCE')
        hdu_map = fits.ImageHDU(maparray.astype(np.int16), name='MAP')
        hdul = fits.HDUList([hdu_combined, hdu_variance, hdu_map])
        hdul.writeto(output_combined, overwrite=True)
        logger.info("Combined (bias subtracted) array, variance, and map saved")

    if args.record:
        log_filename = 'terminal_output.txt'
        with open(log_filename, 'wt') as f:
            f.write(console.export_text(styles=True))
        logger.info("terminal output recorded in %s", log_filename)


if __name__ == "__main__":

    main()
