#
# Copyright 2016-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy


def detrend(arr, x=None, deg=5, tol=1e-3, maxloop=10):
    """
    Compute a baseline trend of a signal

    Parameters
    ----------
    arr
    x
    deg
    tol
    maxloop

    Returns
    -------

    """

    xx = numpy.arange(len(arr)) if x is None else x
    base = arr.copy()
    trend = base
    pol = numpy.ones((deg + 1,))
    for _ in range(maxloop):
        pol_new = numpy.polyfit(xx, base, deg)
        pol_norm = numpy.linalg.norm(pol)
        diff_pol_norm = numpy.linalg.norm(pol - pol_new)
        if diff_pol_norm / pol_norm < tol:
            break
        pol = pol_new
        trend = numpy.polyval(pol, xx)
        base = numpy.minimum(base, trend)
    return trend