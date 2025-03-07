#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Remove extension from FITS file"""
import argparse
from astropy.io import fits


def remove_extension(filename, extension, verbose=True):
    """Remove extension from FITS file

    Parameters
    ----------
    filename : str, file-like or `pathlib.Path`
        FITS filename to be updated.
    extension : str or int
        Extension to remove from FITS file.
        The PRIMARY extension cannot be removed.
    verbose : bool
        If True, display additional information.
    """
    hdul = fits.open(filename, mode='update')
    num_extensions = len(hdul)
    if verbose:
        print(hdul.info())
    if isinstance(extension, int):
        # remove extension by index
        if extension == 0:
            raise ValueError("PRIMARY extension cannot be removed")
        elif extension < len(hdul):
            if verbose:
                print(f"Removing {extension} {hdul[extension].name} from {filename}")
            del hdul[extension]  # programming tip: using hdul.pop(extension) does not work!
        else:
            raise ValueError(f"{extension} is greater than number of available extensions: {num_extensions}")
    elif isinstance(extension, str):
        # remove extension by name
        extname = extension.upper()
        extension_exists = extname in [hdu.name for hdu in hdul]
        if extension_exists:
            for i, hdu in enumerate(hdul):
                if hdu.name == extname:
                    if verbose:
                        print(f"Removing {i} {hdul[extension].name} from {filename}")
                    del hdul[i]   # programming tip: using hdul.pop(i) does not work!
                    break
        else:
            raise ValueError(f"Extension '{extname}' not found")
    else:
        raise ValueError(f"{extension} of type {type(extension)} is not a valid extension")

    if verbose:
        print("Flushing the changes...")

    hdul.flush(output_verify='fix+warn', verbose=verbose)

    if verbose:
        print(hdul.info())

    hdul.close(verbose=verbose)


def main(args=None):
    parser = argparse.ArgumentParser(description="Remmove extension from FITS file")
    parser.add_argument("input", help="Input FITS file", type=str)
    parser.add_argument("--extnum", help="extension number (0=PRIMARY)", type=int, default=None)
    parser.add_argument("--extname", help="extension name", type=str, default=None)
    parser.add_argument("--noverbose", help="do not display additional information", action="store_true")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.extnum is None:
        if args.extname is None:
            with fits.open(args.input) as hdul:
                print(hdul.info())
            raise ValueError("You must specify at least one of --extnum and --extname")
        else:
            extension = args.extname
    else:
        if args.extname is None:
            extension = args.extnum
        else:
            raise ValueError(f"{args.extnum} and {args.extname} cannot be used together")

    verbose = not args.noverbose
    remove_extension(args.input, extension, verbose)


if __name__ == "__main__":
    main()
