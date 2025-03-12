#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Store HDU from different files into a single FITS file with extensions.
"""
import argparse
from astropy.io import fits
import sys

from .ctext import ctext


def stack_hdus(input_list, output_filename, verbose=False):
    """Store HDU from different files into a single FITS file with extensions.

    Parameters
    ----------
    input_list : str
        Input text file with the name of the FITS files
        whose HDU will be stacked together in a single FITS file.
    output_filename : str
        Output FITS file with the stacked HDUs.
    verbose : bool
        If True, display additional information.
    """

    # read input file with list of input FITS images
    with open(input_list) as f:
        file_content = f.read().splitlines()

    hdul_stack = fits.HDUList()
    nstack = 0
    for line in file_content:
        if len(line) > 0:
            if line[0] not in ['#']:
                file_and_extension = line.split(',')
                fname = file_and_extension[0]
                if len(file_and_extension) == 1:
                    extname = 'PRIMARY'
                elif len(file_and_extension) == 2:
                    extname = file_and_extension[1].upper()
                else:
                    raise ValueError(f"Unexpected file, extension: {file_and_extension}")
                if verbose:
                    print(f'* Reading: {fname} -> HDU: {extname}')
                with fits.open(fname) as hdul:
                    if extname not in hdul:
                        raise ValueError(f'Expected {extname} extension not found')
                    data = hdul[extname].data
                    if nstack == 0:
                        hdu = fits.PrimaryHDU(data)
                    else:
                        hdu = fits.ImageHDU(data)
                    hdul_stack.append(hdu)

    if verbose:
        print(f"Saving {output_filename}")
    hdul_stack.writeto(output_filename, overwrite=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="TXT file with list of images whose HDU will be stacked", type=str)
    parser.add_argument('output_filename',
                        help='filename of output FITS image', type=str)

    parser.add_argument("--verbose",
                        help="Display intermediate information",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    input_list = args.input_list
    output_filename = args.output_filename
    verbose = args.verbose

    stack_hdus(input_list, output_filename, verbose)


if __name__ == '__main__':
    main()