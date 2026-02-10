#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute Atmospheric Differential Refraction vs wavelength"""

import argparse
from astropy.table import QTable
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich_argparse import RichHelpFormatter
import sys


def air_refractive_index_15_760(wave_vacuum):
    """Air refractive index (dry air at sea level)

    Equation (1) from Filippenko (1982): refractive index of dry air
    at sea level (P=760 mm Hg, T=15 degree Celsius).

    Parameters
    ----------
    wave_vacuum : `~astropy.units.Quantity`
        Wavelength (vacuum).

    Returns
    -------
    n_air : float
        Refractive index of dray air.

    """

    wave_vacuum_micron = wave_vacuum.to(u.micron).value

    n_air = 64.328 + 29498.1 / (146 - (1 / wave_vacuum_micron) ** 2) + 255.4 / (41 - (1 / wave_vacuum_micron) ** 2)
    n_air = 1 + n_air * 1E-6

    return n_air


def air_refractive_index(wave_vacuum, temperature, pressure_mm, pressure_water_vapor_mm):
    """Air refractive index (general case)

    Equations (2) and (3) from Filippenko (1982).

    Parameters
    ----------
    wave_vacuum : `~astropy.units.Quantity`
        Wavelength (vacuum).
    temperature : `~astropy.units.Quantity`
        Temperature.
    pressure_mm : float
        Pressure (mm Hg).
    pressure_water_vapor_mm : float
        Water vapour pressure (mm Hg).

    Returns
    -------
    n_air : float
        Refractive index of air.

    """

    wave_vacuum_micron = wave_vacuum.to(u.micron).value
    temperature_value = temperature.to(u.Celsius).value

    n_air = (pressure_mm * (1 + (1.049 - 0.0157 * temperature_value) * 1E-6 * pressure_mm)) / \
            (720.833 * (1 + 0.003661 * temperature_value))
    n_air = 1 + (air_refractive_index_15_760(wave_vacuum) - 1) * n_air

    water_factor = (0.0624 - 0.000680/wave_vacuum_micron**2) / \
                   (1 + 0.003661 * temperature_value) * pressure_water_vapor_mm
    n_air -= water_factor * 1E-6

    return n_air


def compute_adr_wavelength(
        airmass,
        reference_wave_vacuum,
        wave_vacuum,
        temperature=7*u.Celsius,
        pressure_mm=600,
        pressure_water_vapor_mm=8,
        debug=False
):
    """Compute differential atmospheric refraction vs wavelength

    Here we employ the same parameters as Filippenko (1982):
    "At an altitude of ~2 km and a latitude of ~ +/-30, average
    conditions are (Allen 1973) P ~ 600 mm Hg, T = 7 Celsius, and
    Pwater ~ 8 mm Hg."

    Parameters
    ----------
    airmass : float
        Airmass.
    reference_wave_vacuum : `~astropy.units.Quantity`
        Reference wavelength to compute the differential refraction
        correction. This wavelength corresponds to a correction of
        zero.
    wave_vacuum : `~astropy.units.Quantity`
        Array containing `nphotons` simulated photons with the
        spectrum requested in the scene block. These values are
        required to compute the differential refraction correction.
        Note that some values in this array could have been set
        to -1 (e.g., removed photons when applying the atmosphere
        transmission).
    temperature : `~astropy.units.Quantity`
        Temperature.
    pressure_mm : float
        Atmospheric pressure (mm Hg).
    pressure_water_vapor_mm : float
        Water vapor pressure (mm Hg).
    debug : bool
        If True, display additional information.

    Returns
    -------
    differential_refraction : `~astropy.units.Quantity`
        Differential refraction at each simulated wavelength.

    """

    if airmass < 1.0:
        raise ValueError(f'Unexpected {airmass=}')

    # zenith distance
    zenith_distance = np.rad2deg(np.arccos(1/airmass)) * u.deg
    if debug:
        print(f'{airmass=} --> {zenith_distance=}')

    # air refractive index for reference wavelength, at the conditions
    # employed by Filippenko (1982)
    n_air_reference = air_refractive_index(
        wave_vacuum=reference_wave_vacuum,
        temperature=temperature,
        pressure_mm=pressure_mm,
        pressure_water_vapor_mm=pressure_water_vapor_mm
    )
    if debug:
        print(f'Assuming {temperature=}, {pressure_mm=}, {pressure_water_vapor_mm=}')
        print(f'{reference_wave_vacuum=}')
        print(f'{n_air_reference=}')

    # air refractive index for all the simulated wavelengths
    n_air = air_refractive_index(
        wave_vacuum=wave_vacuum,
        temperature=temperature,
        pressure_mm=pressure_mm,
        pressure_water_vapor_mm=pressure_water_vapor_mm
    )

    # refraction (plane-parallel atmosphere)
    factor_arcsec_per_radian = 206264.80624709636
    refraction_reference = (n_air_reference - 1) * np.tan(zenith_distance)
    if debug:
        print(f'Refraction at reference wavelength (arcsec): ' +
              f'{refraction_reference * factor_arcsec_per_radian:+.4f}')
    refraction = (n_air - 1) * np.tan(zenith_distance)
    differential_refraction = (refraction - refraction_reference) * factor_arcsec_per_radian * u.arcsec
    if debug:
        # avoid negative wavelengths: those are flagged values corresponding
        # to removed photons (e.g., due to the atmosphere transmission)
        iok = np.argwhere(wave_vacuum > 0 * u.m)
        differential_refraction_ok = differential_refraction[iok]
        simulated_wave_ok = wave_vacuum[iok]
        imin = np.argmin(differential_refraction_ok)
        imax = np.argmax(differential_refraction_ok)
        print(f'Minimum differential refraction: {differential_refraction_ok[imin][0]:+.4f} ' +
              f'at wavelength: {simulated_wave_ok[imin][0]}')
        print(f'Maximum differential refraction: {differential_refraction_ok[imax][0]:+.4f} ' +
              f'at wavelength: {simulated_wave_ok[imax][0]}')

    return differential_refraction


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(description="Compute Atmospheric Differential Refraction",
                                     formatter_class=RichHelpFormatter)
    parser.add_argument("--airmass", help="Airmass", type=float)
    parser.add_argument("--reference_wave_vacuum", help="Reference wavelength (vacuum)", type=float)
    parser.add_argument("--wave_ini", help="Initial wavelength", type=float)
    parser.add_argument("--wave_end", help="Final wavelength", type=float)
    parser.add_argument("--wave_step", help="Wavelength step", type=float)
    parser.add_argument("--wave_unit", help="Wavelength unit (astropy)", type=str)
    parser.add_argument("--temperature", help="Temperature (Celsius)", type=float, default=7)
    parser.add_argument("--pressure", help="Pressure (mmHg)", type=float, default=600)
    parser.add_argument("--pressure_water_vapor", help="Water vapor pressure (mmHg)", type=float, default=8)
    parser.add_argument("--plots", help="Plot intermediate results", action="store_true")
    parser.add_argument("--ndecimal_wave", help="Number of decimal places in wavelength", type=int, default=3)
    parser.add_argument("--ndecimal_adr", help="Number of decimal places in ADR", type=int, default=3)
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    parser.add_argument("--debug", help="Debug", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.debug:
        for arg, value in vars(args).items():
            print(f'--{arg} {value}')

    if args.echo:
        print('[bold red]Executing:\n' + ' '.join(sys.argv) + '[/bold red]')

    if args.wave_ini is None:
        raise ValueError('You must specify --wave_ini')
    if args.wave_end is None:
        raise ValueError('You must specify --wave_end')
    if args.wave_step is None:
        raise ValueError('You must specify --wave_step')

    if args.wave_unit is None:
        raise ValueError('You must specify --wave_unit')
    else:
        try:
            wave_unit = u.Unit(args.wave_unit)
        except ValueError:
            raise ValueError(f'{args.wave_unit} is not a valid unit')

        if not wave_unit.is_equivalent(u.m):
            raise ValueError(f'{args.wave_unit} is not a valid wavelength unit')

        wave_vacuum = np.arange(
            start=args.wave_ini,
            stop=args.wave_end + args.wave_step / 2,
            step=args.wave_step
        ) * wave_unit

    if args.reference_wave_vacuum is None:
        reference_wave_vacuum = (wave_vacuum[0] + wave_vacuum[-1]) / 2
        print(f'Using reference_wave_vacuum={reference_wave_vacuum}')
    else:
        reference_wave_vacuum = args.reference_wave_vacuum * wave_unit

    differential_refraction = compute_adr_wavelength(
        airmass=args.airmass,
        reference_wave_vacuum=reference_wave_vacuum,
        wave_vacuum=wave_vacuum,
        temperature=args.temperature * u.Celsius,
        pressure_mm=args.pressure,
        pressure_water_vapor_mm=args.pressure_water_vapor,
        debug=args.debug
    )

    result = QTable()
    result['Wavelength'] = wave_vacuum
    result['Wavelength'].info.format = f'.{args.ndecimal_wave}f'
    result['ADR'] = differential_refraction
    result['ADR'].info.format = f'.{args.ndecimal_adr}f'
    result.pprint_all()

    if args.plots:
        fig, ax = plt.subplots()
        ax.plot(wave_vacuum, differential_refraction, '.')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel(f'Atmospheric Differential Refraction ({differential_refraction.unit})')
        ax.set_title(f'Airmass: {args.airmass}, reference wave: {reference_wave_vacuum}\n' +
                     f'Temperature: {args.temperature} deg Celsius, Pressure: {args.pressure} mmHg\n' +
                     f'Water Vapor Pressure: {args.pressure_water_vapor} mmHg')
        ax.axhline(0, linestyle='--', color='grey')
        ax.axvline(reference_wave_vacuum.value, linestyle=':', color='C1')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    main()
