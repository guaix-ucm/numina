#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute world_to_pixel.
"""
import argparse
from argparse import RawTextHelpFormatter
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import sys

from .ctext import ctext


def world_to_pixel(inputfile, sky, wave, extnum, verbose=False):
    """Compute world_to_pixel.

    Parameters
    ----------
    inputfile : str, file-like or `pathlib.Path`
        Input FITS filename.
    sky : str or None
        String defining the celestial coordinate with units.
    wave : str or None
        String defining the wavelength with units.
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

    if sky is None or sky == "":
        if wcs.has_celestial:
            sky = wcs.celestial.wcs.crval
            cunit = wcs.celestial.wcs.cunit
            sky = SkyCoord(ra=sky[0]*Unit(cunit[0]), dec=sky[1]*Unit(cunit[1]))
        else:
            sky = None
    else:
        sky = eval(sky)

    if wave is None or wave == "":
        if wcs.has_spectral:
            wave = wcs.spectral.wcs.crval * Unit(wcs.spectral.wcs.cunit[0])
        else:
            wave = None
    else:
        wave = eval(wave)

    print(f'\nComputing pixel for world coordinate {sky}, {wave}:')

    if wcs.has_celestial and wcs.has_spectral:
        result = wcs.world_to_pixel(sky, wave)
    elif wcs.has_celestial:
        result = wcs.world_to_pixel(sky)
    elif wcs.has_spectral:
        result = wcs.world_to_pixel(wave)
    else:
        result = None
    if result is not None:
        result = [item + 1 for item in result]

    output = ''
    for item in result:
        if isinstance(item, np.ndarray):
            output += f"{str(item[0])} "
        else:
            output += f"{str(item)} "
    print(output)


def main(args=None):
    """
    Usage example:
    $ numina-world_to_pixel file.fits \
      --sky 'SkyCoord(0*u.arcsec, 0*u.arcsec)'
      --wave '1.9344e-06*u.m'

    The sky and wavelength coordinates must be provided with units.
    """
    parser = argparse.ArgumentParser(
        description="Convert world to pixel coordinates.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("inputfile", help="Input FITS file", type=str)
    parser.add_argument("--sky", help="Celestial coordinate (string)\n"
                        "e.g. 'SkyCoord(0*u.arcsec, 0*u.arcsec)'", type=str, default=None)
    parser.add_argument("--wave", help="Spectral wavelength (string)\n"
                        "e.g. '1.9344e-06*u.m'", type=str, default=None)
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
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    extnum = args.extnum
    if extnum < 0:
        raise ValueError('extnum must be >= 0')

    world_to_pixel(
        inputfile=args.inputfile,
        sky=args.sky,
        wave=args.wave,
        extnum=extnum,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
