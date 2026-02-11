#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import matplotlib.pyplot as plt
import numpy as np


def fapply_atmosphere_transmission(
    simulated_wave, wave_transmission, curve_transmission, rng, plots=False, verbose=False
):
    """Apply atmosphere transmission.

    The input wavelength of each photon is converted into -1
    if the photon is absorbed. These photons are discarded later
    when the code removes those outside [wmin, vmax].

    Parameters
    ----------
    simulated_wave : `~astropy.units.Quantity`
        Array containing the simulated wavelengths. If the photon is
        absorbed, the wavelength is changed to -1. Note that this
        input array is also the output of this function.
    wave_transmission : `~astropy.units.Quantity`
        Wavelength column of the tabulated transmission curve.
    curve_transmission : `~astropy.units.Quantity`
        Transmission values for the wavelengths given in
        'wave_transmission'.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    plots : bool
        If True, plot input and output results.
    verbose : bool
        If True, display additional information.

    """

    wave_unit = simulated_wave.unit

    # compute transmission at the wavelengths of the simulated photons
    transmission_values = np.interp(
        x=simulated_wave.value, xp=wave_transmission.to(wave_unit).value, fp=curve_transmission
    )

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave_transmission.to(wave_unit), curve_transmission, "-", label="SKYCALC curve")
        ax.plot(simulated_wave, transmission_values, ",", alpha=0.5, label="interpolated values")
        ax.set_xlabel(f"Wavelength ({wave_unit})")
        ax.set_ylabel("Transmission fraction")
        ax.legend(loc=3)  # loc="best" can be slow with large amounts of data
        plt.tight_layout()
        plt.show()

    # generate random values in the interval [0, 1] and discard photons whose
    # transmission value is lower than the random value
    nphotons = len(simulated_wave)
    survival_probability = rng.uniform(low=0, high=1, size=nphotons)
    iremove = np.argwhere(transmission_values < survival_probability)
    simulated_wave[iremove] = -1 * wave_unit

    if verbose:
        print("Applying atmosphere transmission:")
        print(f"- initial number of photons: {nphotons}")
        textwidth_nphotons_number = len(str(nphotons))
        percentage = np.round(100 * len(iremove) / nphotons, 2)
        print(f"- number of photons removed: {len(iremove): > {textwidth_nphotons_number}}  ({percentage}%)")
