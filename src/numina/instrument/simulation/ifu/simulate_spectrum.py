#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.constants as constants
import astropy.units as u
import logging
import matplotlib.pyplot as plt
import numpy as np

from .raise_valueerror import raise_ValueError


def simulate_spectrum(
    wave,
    flux,
    flux_type,
    nphotons,
    wavelength_sampling,
    rng,
    wmin,
    wmax,
    convolve_sigma_km_s,
    nbins_histo,
    plots,
    plot_title,
    logger=None,
):
    """Simulate spectrum defined by tabulated wave and flux data.

    Parameters
    ----------
    wave : `~astropy.units.Quantity`
        Numpy array (with astropy units) containing the tabulated
        wavelength.
    flux : array_like
        Array-like object containing the tabulated flux.
    flux_type : str
        Relative flux unit. Valid options are:
        - flam: proportional to erg s^-1 cm^-2 A^-1
        - photlam: proportional to photon s^-1 cm^-2 A^-1
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
    convolve_sigma_km_s : `~astropy.units.Quantity`
        Gaussian broadening (sigma) in km/s to be applied.
    nbins_histo : int
        Number of bins for histogram plot.
    plots : bool
        If True, plot input and output results.
    plot_title : str or None
        Plot title. Used only when 'plots' is True.
    logger : logging.Logger or None, optional
        Logger for logging messages. If None, a default logger will be used.

    Returns
    -------
    simulated_wave : `~astropy.units.Quantity`
        Wavelength of simulated photons.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    flux = np.asarray(flux)
    if len(wave) != len(flux):
        raise_ValueError(f"Incompatible array length: 'wave' ({len(wave)}), 'flux' ({len(flux)})")

    if np.any(flux < 0):
        raise_ValueError(f"Negative flux values cannot be handled")

    if flux_type.lower() not in ["flam", "photlam"]:
        raise_ValueError(f"Flux type: {flux_type} is not any of the valid values: 'flam', 'photlam'")

    if not isinstance(wave, u.Quantity):
        raise_ValueError(f"Object {wave=} is not a Quantity instance")
    wave_unit = wave.unit
    if not wave_unit.is_equivalent(u.m):
        raise_ValueError(f"Unexpected unit for 'wave': {wave_unit}")

    # lower wavelength limit
    if wmin is not None:
        if not isinstance(wmin, u.Quantity):
            raise_ValueError(f"Object {wmin=} is not a Quantity instance")
        if not wmin.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmin': {wmin}")
        wmin = wmin.to(wave_unit)
        lower_index = np.searchsorted(wave.value, wmin.value, side="left")
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise_ValueError(f"Object {wmax=} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmax': {wmin}")
        wmax = wmax.to(wave_unit)
        upper_index = np.searchsorted(wave.value, wmax.value, side="right")
    else:
        upper_index = len(wave)

    if lower_index == upper_index:
        if plot_title is not None:
            print(f"Working with data from: {plot_title}")
        print(f"Tabulated wavelength range: {wave[0]} - {wave[-1]}")
        print(f"Requested wavelength range: {wmin} - {wmax}")
        raise_ValueError("Wavelength ranges without intersection")

    if not isinstance(convolve_sigma_km_s, u.Quantity):
        raise_ValueError(f"Object {convolve_sigma_km_s=} is not a Quantity instance")
    if convolve_sigma_km_s.unit != u.km / u.s:
        raise_ValueError(f"Unexpected unit for {convolve_sigma_km_s}")
    if convolve_sigma_km_s.value < 0:
        raise_ValueError(f"Unexpected negative value for {convolve_sigma_km_s}")

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave.value, flux, "-")
        if wmin is not None:
            ax.axvline(wmin.value, linestyle="--", color="gray")
        if wmax is not None:
            ax.axvline(wmax.value, linestyle="--", color="gray")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Flux (arbitrary units)")
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    wave = wave[lower_index:upper_index]
    flux = flux[lower_index:upper_index]

    # convert FLAM to PHOTLAM
    if flux_type.lower() == "flam":
        logger.debug("Converting FLAM to PHOTLAM")
        flux_conversion = wave.to(u.m) / (constants.h * constants.c)
        flux *= flux_conversion.value

    wmin_eff = wave[0]
    wmax_eff = wave[-1]

    # normalized cumulative area
    # (area under the polygons defined by the tabulated data)
    cumulative_area = np.concatenate(([0], np.cumsum((flux[:-1] + flux[1:]) / 2 * (wave[1:] - wave[:-1]))))
    normalized_cumulative_area = cumulative_area / cumulative_area[-1]

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave.value, normalized_cumulative_area, ".")
        ax.axvline(wmin_eff.value, linestyle="--", color="gray")
        ax.axvline(wmax_eff.value, linestyle="--", color="gray")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Normalized cumulative area")
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

    simulated_wave = np.interp(x=unisamples, xp=normalized_cumulative_area, fp=wave.value)

    # apply Gaussian broadening
    if convolve_sigma_km_s.value > 0:
        logger.debug(f"Applying {convolve_sigma_km_s=}")
        sigma_wave = convolve_sigma_km_s / constants.c.to(u.km / u.s) * simulated_wave
        simulated_wave = rng.normal(loc=simulated_wave, scale=sigma_wave)

    # add units
    simulated_wave *= wave_unit

    if plots:
        fig, ax = plt.subplots()
        hist_sim, bin_edges_sim = np.histogram(simulated_wave.value, bins=nbins_histo)
        xhist_sim = (bin_edges_sim[:-1] + bin_edges_sim[1:]) / 2
        fscale = np.median(hist_sim / np.interp(x=xhist_sim, xp=wave.value, fp=flux))
        ax.plot(wave.value, flux * fscale, "k-", linewidth=1, label="rescaled input spectrum")
        hist_dum = np.diff(np.interp(x=bin_edges_sim, xp=wave.value, fp=normalized_cumulative_area)) * nphotons
        ax.plot(xhist_sim, hist_dum, "-", linewidth=3, label="binned input spectrum")
        ax.plot(xhist_sim, hist_sim, "-", linewidth=1, label="binned simulated spectrum")
        ax.axvline(wmin_eff.value, linestyle="--", color="gray")
        ax.axvline(wmax_eff.value, linestyle="--", color="gray")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Number of simulated photons")
        if plot_title is not None:
            ax.set_title(plot_title)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return simulated_wave
