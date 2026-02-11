#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.units as u
import numpy as np

from .raise_valueerror import raise_ValueError


def simulate_constant_photlam(wmin, wmax, nphotons, wavelength_sampling, rng):
    """Simulate spectrum with constant flux (in PHOTLAM units).

    Parameters
    ----------
    wmin : `~astropy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength to be considered.
    nphotons : int
        Number of photons to be simulated.
    wavelength_sampling : str
        Method to sample the wavelength values. Two options are valid:
        - 'random': the wavelengt of each photon is randomly determined
          using the spectrum shape as the density probability function.
        - 'fixed': the wavelength of each photon is exactly determined
          using the spectrum shape as the density probability function.
    rng : `~numpy.random._generator.Generator`
        Random number generator.

    """

    if not isinstance(wmin, u.Quantity):
        raise_ValueError(f"Object 'wmin': {wmin} is not a Quantity instance")
    if not isinstance(wmax, u.Quantity):
        raise_ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
    if wmin.unit != wmax.unit:
        raise_ValueError(
            f"Different units used for 'wmin' and 'wmax': {wmin.unit}, {wmax.unit}.\n"
            + "Employ the same unit to unambiguously define the output result."
        )

    if wavelength_sampling == "random":
        simulated_wave = rng.uniform(low=wmin.value, high=wmax.value, size=nphotons)
    elif wavelength_sampling == "fixed":
        simulated_wave = np.linspace(wmin.value, wmax.value, num=nphotons)
    else:
        simulated_wave = None  # avoid PyCharm warning
        raise_ValueError(f"Unexpected {wavelength_sampling=}")

    simulated_wave *= wmin.unit
    return simulated_wave
