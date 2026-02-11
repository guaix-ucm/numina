#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from .raise_valueerror import raise_ValueError


def simulate_delta_lines(
    line_wave, line_flux, nphotons, wavelength_sampling, rng, wmin=None, wmax=None, plots=False, plot_title=None
):
    """Simulate spectrum defined from isolated wavelengths.

    Parameters
    ----------
    line_wave : `~astropy.units.Quantity`
        Numpy array (with astropy units) containing the individual
        wavelength of each line.
    line_flux : array_like
        Array-like object containing the individual flux of each line.
    nphotons : int
        Number of photons to be simulated
    wavelength_sampling : str
        Method to sample the wavelength values. Two options are valid:
        - 'random': the wavelengt of each photon is randomly determined
          using the spectrum shape as the density probability function.
        - 'fixed': the wavelength of each photon is exactly determined
          using the spectrum shape as the density probability function.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    wmin : `~astropy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength to be considered.
    plots : bool
        If True, plot input and output results.
    plot_title : str or None
        Plot title. Used only when 'plots' is True.

    Returns
    -------
    simulated_wave : `~astropy.units.Quantity`
        Wavelength of simulated photons.

    """

    line_flux = np.asarray(line_flux)
    if len(line_wave) != len(line_flux):
        raise_ValueError(f"Incompatible array length: 'line_wave' ({len(line_wave)}), 'line_flux' ({len(line_flux)})")

    if np.any(line_flux < 0):
        raise_ValueError(f"Negative line fluxes cannot be handled")

    if not isinstance(line_wave, u.Quantity):
        raise_ValueError(f"Object 'line_wave': {line_wave} is not a Quantity instance")
    wave_unit = line_wave.unit
    if not wave_unit.is_equivalent(u.m):
        raise_ValueError(f"Unexpected unit for 'line_wave': {wave_unit}")

    # lower wavelength limit
    if wmin is not None:
        if not isinstance(wmin, u.Quantity):
            raise_ValueError(f"Object 'wmin': {wmin} is not a Quantity instance")
        if not wmin.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmin': {wmin}")
        wmin = wmin.to(wave_unit)
        lower_index = np.searchsorted(line_wave.value, wmin.value, side="left")
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise_ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmax': {wmin}")
        wmax = wmax.to(wave_unit)
        upper_index = np.searchsorted(line_wave.value, wmax.value, side="right")
    else:
        upper_index = len(line_wave)

    if plots:
        fig, ax = plt.subplots()
        ax.stem(line_wave.value, line_flux, markerfmt=" ", basefmt=" ")
        if wmin is not None:
            ax.axvline(wmin.value, linestyle="--", color="gray")
        if wmax is not None:
            ax.axvline(wmax.value, linestyle="--", color="gray")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Intensity (arbitrary units)")
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    line_wave = line_wave[lower_index:upper_index]
    line_flux = line_flux[lower_index:upper_index]

    # normalized cumulative sum
    cumsum = np.cumsum(line_flux)
    cumsum /= cumsum[-1]

    if plots:
        fig, ax = plt.subplots()
        ax.plot(line_wave.value, cumsum, "-")
        if wmin is not None:
            ax.axvline(wmin.value, linestyle="--", color="gray")
        if wmax is not None:
            ax.axvline(wmax.value, linestyle="--", color="gray")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Cumulative sum")
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    if wavelength_sampling == "random":
        # samples following a uniform distribution
        unisamples = rng.uniform(low=0, high=1, size=nphotons)
    elif wavelength_sampling == "fixed":
        # constant sampling of distribution function
        unisamples = np.linspace(0, 1, num=nphotons + 1)  # generate extra photon
        unisamples = unisamples[:-1]  # remove last photon
    else:
        unisamples = None  # avoid PyCharm warning
        raise_ValueError(f"Unexpected {wavelength_sampling=}")

    # closest array indices in sorted array
    closest_indices = np.searchsorted(cumsum, unisamples, side="right")

    # simulated wavelengths
    simulated_wave = line_wave.value[closest_indices]
    simulated_wave *= wave_unit

    if plots:
        # count number of photons at each tabulated wavelength value
        x_spectrum, y_spectrum = np.unique(simulated_wave, return_counts=True)

        # scale factor to overplot expected spectrum with same total number
        # of photons as the simulated dataset
        factor = np.sum(line_flux) / nphotons

        # overplot expected and simulated spectrum
        fig, ax = plt.subplots()
        ax.stem(line_wave.value, line_flux / factor, markerfmt=" ", basefmt=" ")
        ax.plot(x_spectrum, y_spectrum, ".")
        if wmin is not None:
            ax.axvline(wmin.value, linestyle="--", color="gray")
        if wmax is not None:
            ax.axvline(wmax.value, linestyle="--", color="gray")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Intensity (number of photons)")
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    return simulated_wave
