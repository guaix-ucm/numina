#
# Copyright 2024-2026 Universidad Complutense de Madrid
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
import logging
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich_argparse import RichHelpFormatter
import sys

from numina.user.console import print_table

from .initialize_script_with_args import include_default_arguments_for_common_actions
from .initialize_script_with_args import initialize_script_with_args
from .initialize_script_with_args import goodbye_message_and_save_console


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
    n_air = 1 + n_air * 1e-6

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

    n_air = (pressure_mm * (1 + (1.049 - 0.0157 * temperature_value) * 1e-6 * pressure_mm)) / (
        720.833 * (1 + 0.003661 * temperature_value)
    )
    n_air = 1 + (air_refractive_index_15_760(wave_vacuum) - 1) * n_air

    water_factor = (
        (0.0624 - 0.000680 / wave_vacuum_micron**2) / (1 + 0.003661 * temperature_value) * pressure_water_vapor_mm
    )
    n_air -= water_factor * 1e-6

    return n_air


def compute_adr_wavelength(
    airmass,
    reference_wave_vacuum,
    wave_vacuum,
    temperature=7 * u.Celsius,
    pressure_mm=600,
    pressure_water_vapor_mm=8,
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

    Returns
    -------
    differential_refraction : `~astropy.units.Quantity`
        Differential refraction at each simulated wavelength.

    """
    logger = logging.getLogger(__name__)

    if airmass < 1.0:
        raise ValueError(f"Unexpected {airmass=}")

    # zenith distance
    zenith_distance = np.rad2deg(np.arccos(1 / airmass)) * u.deg
    logger.info(f"{airmass=} --> {zenith_distance=}")

    # air refractive index for reference wavelength, at the conditions
    # employed by Filippenko (1982)
    n_air_reference = air_refractive_index(
        wave_vacuum=reference_wave_vacuum,
        temperature=temperature,
        pressure_mm=pressure_mm,
        pressure_water_vapor_mm=pressure_water_vapor_mm,
    )
    logger.info(f"Assuming {temperature=}, {pressure_mm=}, {pressure_water_vapor_mm=}")
    logger.info(f"{reference_wave_vacuum=}")
    logger.info(f"{n_air_reference=}")

    # air refractive index for all the simulated wavelengths
    n_air = air_refractive_index(
        wave_vacuum=wave_vacuum,
        temperature=temperature,
        pressure_mm=pressure_mm,
        pressure_water_vapor_mm=pressure_water_vapor_mm,
    )

    # refraction (plane-parallel atmosphere)
    factor_arcsec_per_radian = 206264.80624709636
    refraction_reference = (n_air_reference - 1) * np.tan(zenith_distance)
    logger.info(
        f"Refraction at reference wavelength (arcsec): " + f"{refraction_reference * factor_arcsec_per_radian:+.4f}"
    )
    refraction = (n_air - 1) * np.tan(zenith_distance)
    differential_refraction = (refraction - refraction_reference) * factor_arcsec_per_radian * u.arcsec

    # avoid negative wavelengths: those are flagged values corresponding
    # to removed photons (e.g., due to the atmosphere transmission)
    iok = np.argwhere(wave_vacuum > 0 * u.m)
    differential_refraction_ok = differential_refraction[iok]
    simulated_wave_ok = wave_vacuum[iok]
    imin = np.argmin(differential_refraction_ok)
    imax = np.argmax(differential_refraction_ok)
    logger.debug(
        f"Minimum differential refraction: {differential_refraction_ok[imin][0]:+.4f} "
        + f"at wavelength: {simulated_wave_ok[imin][0]}"
    )
    logger.debug(
        f"Maximum differential refraction: {differential_refraction_ok[imax][0]:+.4f} "
        + f"at wavelength: {simulated_wave_ok[imax][0]}"
    )

    return differential_refraction


def main(args=None):
    """Main function"""

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="Compute Atmospheric Differential Refraction", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--airmass", help="Airmass", type=float)
    parser.add_argument("--reference-wave-vacuum", help="Reference wavelength (vacuum)", type=float)
    parser.add_argument("--wave-ini", help="Initial wavelength", type=float)
    parser.add_argument("--wave-end", help="Final wavelength", type=float)
    parser.add_argument("--wave-step", help="Wavelength step", type=float)
    parser.add_argument("--wave-unit", help="Wavelength unit (astropy)", type=str)
    parser.add_argument("--temperature", help="Temperature (Celsius)", type=float, default=7)
    parser.add_argument("--pressure", help="Pressure (mmHg)", type=float, default=600)
    parser.add_argument("--pressure_water_vapor", help="Water vapor pressure (mmHg)", type=float, default=8)
    parser.add_argument("--plots", help="Plot intermediate results", action="store_true")
    parser.add_argument("--ndecimal-wave", help="Number of decimal places in wavelength", type=int, default=3)
    parser.add_argument("--ndecimal-adr", help="Number of decimal places in ADR", type=int, default=3)
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args=args)

    # Initialize the script with the provided arguments
    console, logger, datetime_ini = initialize_script_with_args(sys.argv, parser, args, __name__)

    if args.wave_ini is None:
        raise ValueError("You must specify --wave-ini")
    if args.wave_end is None:
        raise ValueError("You must specify --wave-end")
    if args.wave_step is None:
        raise ValueError("You must specify --wave-step")

    if args.wave_unit is None:
        raise ValueError("You must specify --wave-unit")
    else:
        try:
            wave_unit = u.Unit(args.wave_unit)
        except ValueError:
            raise ValueError(f"{args.wave_unit} is not a valid unit")

        if not wave_unit.is_equivalent(u.m):
            raise ValueError(f"{args.wave_unit} is not a valid wavelength unit")

        wave_vacuum = (
            np.arange(start=args.wave_ini, stop=args.wave_end + args.wave_step / 2, step=args.wave_step) * wave_unit
        )

    if args.reference_wave_vacuum is None:
        reference_wave_vacuum = (wave_vacuum[0] + wave_vacuum[-1]) / 2
        print(f"Using reference-wave-vacuum={reference_wave_vacuum}")
    else:
        reference_wave_vacuum = args.reference_wave_vacuum * wave_unit

    differential_refraction = compute_adr_wavelength(
        airmass=args.airmass,
        reference_wave_vacuum=reference_wave_vacuum,
        wave_vacuum=wave_vacuum,
        temperature=args.temperature * u.Celsius,
        pressure_mm=args.pressure,
        pressure_water_vapor_mm=args.pressure_water_vapor,
    )

    result = QTable()
    result["Wavelength"] = wave_vacuum
    result["Wavelength"].info.format = f".{args.ndecimal_wave}f"
    result["ADR"] = differential_refraction
    result["ADR"].info.format = f".{args.ndecimal_adr}f"
    # result.pprint_all()  # this is not recordable in the console
    print_table(console, result, header_style="red")

    if args.plots:
        fig, ax = plt.subplots()
        ax.plot(wave_vacuum, differential_refraction, ".")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel(f"Atmospheric Differential Refraction ({differential_refraction.unit})")
        ax.set_title(
            f"Airmass: {args.airmass}, reference wave: {reference_wave_vacuum}\n"
            + f"Temperature: {args.temperature} deg Celsius, Pressure: {args.pressure} mmHg\n"
            + f"Water Vapor Pressure: {args.pressure_water_vapor} mmHg"
        )
        ax.axhline(0, linestyle="--", color="grey")
        ax.axvline(reference_wave_vacuum.value, linestyle=":", color="C1")
        plt.tight_layout()
        plt.show()

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":

    main()
