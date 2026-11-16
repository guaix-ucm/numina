#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""
Logarithm of a spectrum, valid also for non-positive values.
"""

import numpy as np


def log_spectrum(sp, percentile=1.0):
    """Logarithm of a spectrum, valid also for non-positive values.

    The spectrum is first clipped from below at the value c of the
    given percentile, and the logarithm is computed as
    log10(max(sp, c) - c + 1). In this way, non-positive values are
    handled properly, and a few strongly negative pixels do not
    compress the dynamic range of the result (they are set to 0).

    Parameters
    ----------
    sp : 1d numpy array
        Spectrum.
    percentile : float
        Percentile (from 0 to 100) of the finite values of 'sp'
        employed as the lower clipping value. If 0, the minimum of
        the spectrum is employed, and no pixel is clipped.

    Returns
    -------
    splog : 1d numpy array
        Logarithm of the spectrum. Non-finite input values remain
        NaN.
    ylabel : str
        Label describing splog, to be used in the plot.

    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100] (percentile={percentile})")

    sp = np.asarray(sp, dtype=float)
    ylabel = f"log10(counts - P{percentile:g} + 1)" if percentile > 0 else "log10(counts - min(counts) + 1)"

    finite = np.isfinite(sp)
    if not np.any(finite):
        return np.full(sp.shape, np.nan), ylabel

    # lower clipping value (computed only with finite values)
    spfloor = np.percentile(sp[finite], percentile)
    splog = np.full(sp.shape, np.nan)
    # after clipping, the argument of log10 is always >= 1
    splog[finite] = np.log10(np.maximum(sp[finite], spfloor) - spfloor + 1.0)
    return splog, ylabel
