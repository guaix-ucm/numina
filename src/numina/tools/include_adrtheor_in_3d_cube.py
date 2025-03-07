#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Include ADR prediction as extension in a 3D FITS file"""

import argparse
from astropy.io import fits
import astropy.units as u
from astropy.units import Unit
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
import numpy as np
import sys

from .compute_adr_wavelength import compute_adr_wavelength
from .compare_adr_extensions_in_3d_cube import compare_adr_extensions_in_3d_cube


def include_adrtheor_in_3d_cube(
        filename,
        extname,
        reference_vacuum_wavelength_angstrom,
        temperature,
        pressure_mm,
        pressure_water_vapor_mm,
        plots,
        verbose=False
):
    """Include ADR prediction as extension in a 3D FITS file

    Parameters
    ----------
    filename : str, file-like or `pathlib.Path`
        FITS filename to be updated.
    extname : str
        Extension to store the ADR prediction in.
    reference_vacuum_wavelength_angstrom : float or None
        Reference wavelength (in Angstrom) to compute ADR prediction.
    verbose : bool
        If True, display additional information.
    """

    with fits.open(filename) as hdul:
        if verbose:
            print(hdul.info())
        header = hdul[0].header

    if header["NAXIS"] != 3:
        raise ValueError("Expected NAXIS=3 not found in PRIMARY HDU")

    # generate 3D WCS object
    wcs3d = WCS(header)
    naxis1, naxis2, naxis3 = wcs3d.pixel_shape
    if verbose:
        print(wcs3d.spectral)

    # array of wavelengths along NAXIS3
    wave = wcs3d.spectral.pixel_to_world(np.arange(naxis3))

    # reference wavelength to compute ADR
    if reference_vacuum_wavelength_angstrom is not None:
        reference_vacuum_wavelength = reference_vacuum_wavelength_angstrom * Unit('Angstrom')
        reference_vacuum_wavelength = reference_vacuum_wavelength.to(Unit('m'))
    else:
        reference_vacuum_wavelength = (wave[0] + wave[-1]) / 2
    if verbose:
        print(f'Reference wavelength: {reference_vacuum_wavelength}')

    # airmass
    if 'AIRMASS' in header:
        airmass = header['AIRMASS']
        if verbose:
            print(f'AIRMASS: {airmass}')
    else:
        raise ValueError('Header does not contain AIRMASS')

    # differential refraction (arcsec)
    differential_refraction = compute_adr_wavelength(
        airmass=airmass,
        reference_wave_vacuum=reference_vacuum_wavelength,
        wave_vacuum=wave,
        temperature=temperature,
        pressure_mm=pressure_mm,
        pressure_water_vapor_mm=pressure_water_vapor_mm,
    )
    if verbose:
        print(f'Differential refraction: {differential_refraction}')

    # parallactic angle
    if 'PARANGLE' in header:
        parangle = Angle(header['PARANGLE'] * Unit('deg'))
        if verbose:
            print(f'PARANGLE: {parangle}')
    else:
        raise ValueError('Header does not contain PARANGLE')

    # predict ADR at the center of the field of view
    x_center_ifu, y_center_ifu = wcs3d.celestial.wcs.crpix    # FITS convention
    center_ifu_coord = wcs3d.celestial.pixel_to_world(
        x_center_ifu - 1.0,   # Python convention
        y_center_ifu - 1.0    # Python convention
    )
    if verbose:
        print(f'Center IFU coord: {center_ifu_coord}')

    # duplicate initial central coordinates at each slice along NAXIS3
    ra_center_ifu = np.repeat(center_ifu_coord.ra, naxis3)
    dec_center_ifu = np.repeat(center_ifu_coord.dec, naxis3)

    # apply differential refraction correction
    dec_center_ifu += differential_refraction.to(u.deg) * np.cos(parangle)
    ra_center_ifu += differential_refraction.to(u.deg) * np.sin(parangle) / np.cos(dec_center_ifu)

    # compute variation in (X,Y) coordinates
    x_center_ifu_corrected, y_center_ifu_corrected = wcs3d.celestial.world_to_pixel(
        SkyCoord(ra=ra_center_ifu, dec=dec_center_ifu)
    )
    x_center_ifu_corrected += 1
    y_center_ifu_corrected += 1
    delta_x_center_ifu = x_center_ifu_corrected - x_center_ifu
    delta_y_center_ifu = y_center_ifu_corrected - y_center_ifu

    # save result in extension
    if extname != 'NONE':
        if verbose:
            print(f'Updating file {filename}')
        # binary table to store result
        col1 = fits.Column(name='Delta_x', format='D', array=delta_x_center_ifu, unit='pixel')
        col2 = fits.Column(name='Delta_y', format='D', array=delta_y_center_ifu, unit='pixel')
        hdu_result = fits.BinTableHDU.from_columns([col1, col2])
        hdu_result.name = extname.upper()
        hdu_result.header['AIRMASS'] = (airmass, 'Airmass')
        hdu_result.header['PARANGLE'] = (parangle.value, 'Parallactic angle (deg)')
        hdu_result.header['REFEWAVE'] = (reference_vacuum_wavelength.to(u.m).value,
                                         'Reference vacuum wavelength (m)')
        hdu_result.header['TEMPERAT'] = (temperature.value, 'Assumed temperature (Celsius degrees)')
        hdu_result.header['PRESSURE'] = (pressure_mm, 'Pressure (Hg mm)')
        hdu_result.header['PRESSUWV'] = (pressure_water_vapor_mm, 'Water vapor pressure (Hg mm)')
        # open and update existing FITS file
        hdul = fits.open(filename, mode='update')
        if extname in hdul:
            if verbose:
                print(f"Updating extension '{extname}'")
            hdul[extname] = hdu_result
        else:
            if verbose:
                print(f"Adding new extension '{extname}'")
            hdul.append(hdu_result)
        hdul.flush()
        hdul.close()

    # display results
    if plots:
        compare_adr_extensions_in_3d_cube(filename, extname1=extname, extname2=None)


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(description="Include ADR prediction as extension in a 3D FITS file")
    parser.add_argument("filename", help="Input 3D FITS file")
    parser.add_argument("--extname",
                        help="Output extension name to store result (default ADRTHEOR)",
                        type=str, default='ADRTHEOR')
    parser.add_argument("--reference_vacuum_wavelength",
                        help="Reference vacuum wavelength (in Angstrom) to compute ADR prediction",
                        type=float, default=None)
    parser.add_argument("--temperature", help="Temperature in degree Celsius",
                        type=float, default=7)
    parser.add_argument("--pressure_mm", help="Pressure in Hg mm",
                        type=float, default=600)
    parser.add_argument("--pressure_water_vapor_mm", help="Pressure water vapor in Hg mm",
                        type=float, default=8)
    parser.add_argument("--plots", help="Plot intermediate results", action="store_true")
    parser.add_argument("--verbose", help="Display intermediate information", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # protections
    extname = args.extname.upper()
    if len(extname) > 8:
        raise ValueError(f"Extension '{extname}' must be less than 9 characters")

    reference_vacuum_wavelength_angstrom = args.reference_vacuum_wavelength
    if reference_vacuum_wavelength_angstrom is not None:
        if reference_vacuum_wavelength_angstrom <= 0.0:
            raise ValueError("Reference vacuum wavelength must be positive")

    include_adrtheor_in_3d_cube(
        filename=args.filename,
        extname=extname,
        reference_vacuum_wavelength_angstrom=reference_vacuum_wavelength_angstrom,
        temperature=args.temperature*u.Celsius,
        pressure_mm=args.pressure_mm,
        pressure_water_vapor_mm=args.pressure_water_vapor_mm,
        plots=args.plots,
        verbose=args.verbose
    )


if __name__ == '__main__':

    main()
