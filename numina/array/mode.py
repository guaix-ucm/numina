#
# -*- coding: utf8 -*-
#
# Copyright 2014-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

'''Mode estimators.'''

import math

import numpy as np


# HSM method
# http://www.davidbickel.com/
def mode_half_sample(a, is_sorted=False):
    """
    Estimate the mode using the Half Sample mode.

    A method to estimate the mode, as described in
    D. R. Bickel and R. Frühwirth (contributed equally),
    "On a fast, robust estimator of the mode: Comparisons to other
    robust estimators with applications,"
    Computational Statistics and Data Analysis 50, 3500-3530 (2006).


    Example
    =======

    >>> import numpy as np
    >>> np.random.seed(1392838)
    >>> a = np.random.normal(1000, 200, size=1000)
    >>> a[:100] = np.random.normal(2000, 300, size=100)
    >>> b = np.sort(a)
    >>> mode_half_sample(b, is_sorted=True)
    1041.9327885039545

    """

    a = np.asanyarray(a)

    if not is_sorted:
        sdata = np.sort(a)
    else:
        sdata = a

    n = len(sdata)
    if n == 1:
        return sdata[0]
    elif n == 2:
        return 0.5 * (sdata[0] + sdata[1])
    elif n == 3:
        ind = -sdata[0] + 2 * sdata[1] - sdata[2]
        if ind < 0:
            return 0.5 * (sdata[0] + sdata[1])
        elif ind > 0:
            return 0.5 * (sdata[1] + sdata[2])
        else:
            return sdata[1]
    else:
        N = int(math.ceil(n / 2.0))
        w = sdata[(N-1):] - sdata[:(n-N+1)]
        ar = w.argmin()
        return mode_half_sample(sdata[ar:ar+N], is_sorted=True)


def mode_sex(a):
    '''Estimate the mode as sextractor'''
    return 2.5 * np.median(a) - 1.5 * np.mean(a)
