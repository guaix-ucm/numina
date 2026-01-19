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


def estimate_diagnostic_limits(rng, gain, rnoise, maxvalue, num_images, npixels):
    """Estimate the limits for the diagnostic plot.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator for reproducibility.
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
        data = rng.poisson(lam=lam * gain).astype(float) / gain
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
