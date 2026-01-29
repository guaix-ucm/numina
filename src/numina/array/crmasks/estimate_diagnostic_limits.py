#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Estimate the limits for the diagnostic plot."""

import numpy as np


def estimate_diagnostic_limits(
    rng, mm_photon_distribution, mm_nbinom_shape, gain, rnoise, maxvalue, num_images, npixels
):
    """Estimate the limits for the diagnostic plot.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    mm_photon_distribution : str
        The type of photon distribution to use for the numerical simulations.
        Valid options are:
        - 'poisson': use a Poisson distribution for the photon noise.
        - 'nbinom': use a negative binomial distribution for the photon noise.
          In this case, the shape parameter must be provided via the
          'mm_nbinom_shape' parameter.
    mm_nbinom_shape : float
        The shape parameter for the negative binomial distribution. Only
        used if 'mm_photon_distribution' is set to 'nbinom'.
    gain : float
        Gain value (in e/ADU) of the detector.
    rnoise : float
        Readout noise (in ADU) of the detector.
    maxvalue : float
        Maximum pixel value (in ADU) of the detector after
        subtracting the bias.
    num_images : int
        Number of different exposures.
    npixels : int
        Number of simulations to perform for each corner of the
        diagnostic plot.

    Returns
    -------
    xdiag_min : float
        Estimated minimum x value for the diagnostic plot.
    xdiag_max : float
        Estimated maximum x value for the diagnostic plot.
    ydiag_min : float
        Estimated minimum y value for the diagnostic plot.
    ydiag_max : float
        Estimated maximum y value for the diagnostic plot.
    """

    if maxvalue < 0:
        maxvalue = 0.0
    xdiag_min = np.zeros(npixels, dtype=float)
    xdiag_max = np.zeros(npixels, dtype=float)
    ydiag_min = np.zeros(npixels, dtype=float)
    ydiag_max = np.zeros(npixels, dtype=float)
    for i in range(npixels):
        # lower limits
        data = rng.normal(loc=0, scale=rnoise, size=num_images)
        min1d = np.min(data)
        median1d = np.median(data)
        xdiag_min[i] = median1d
        ydiag_min[i] = median1d - min1d
        # upper limits
        lam = np.array([maxvalue] * num_images)
        if mm_photon_distribution == "poisson":
            data = rng.poisson(lam=lam * gain).astype(float) / gain
        else:
            data = (
                rng.negative_binomial(
                    n=mm_nbinom_shape,
                    p=mm_nbinom_shape / (mm_nbinom_shape + lam * gain),
                ).astype(float)
                / gain
            )
        if rnoise > 0:
            data += rng.normal(loc=0, scale=rnoise, size=num_images)
        min1d = np.min(data)
        median1d = np.median(data)
        xdiag_max[i] = median1d
        ydiag_max[i] = median1d - min1d
    xdiag_min = np.min(xdiag_min)
    ydiag_min = np.min(ydiag_min)
    xdiag_max = np.max(xdiag_max)
    ydiag_max = np.max(ydiag_max)
    return xdiag_min, xdiag_max, ydiag_min, ydiag_max
