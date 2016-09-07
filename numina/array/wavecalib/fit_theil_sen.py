#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import division
from __future__ import print_function

import numpy as np


def fit_theil_sen(x, y):
    """Compute a robust linear fit using the Theil-Sen method.

    See http://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator for
    details. This function "pairs up sample points by the rank of
    their x-coordinates (the point with the smallest coordinate being
    paired with the first point above the median coordinate, etc.) and
    computes the median of the slopes of the lines determined by these
    pairs of points".

    Parameters
    ----------
    x : 1d numpy array, float
        X coordinate.
    y : 1d numpy array, float
        Y coordinate.

    Returns
    -------
    intercept : float
        Intercept of the linear fit.
    slope : float
        Slope of the linear fit.

    """

    if x.ndim == y.ndim == 1:
        n = x.size
        if n == y.size:
            if n < 5:
                raise ValueError('n=' + str(n) + ' is < 5')
            result = []  # python list
            if (n % 2) == 0:
                iextra = 0
            else:
                iextra = 1
            for i in range(n//2):
                ii = i + n//2 + iextra
                deltax = x[ii]-x[i]
                deltay = y[ii]-y[i]
                result.append(deltay/deltax)
            slope = np.median(result)
            result = y - slope*x  # numpy array
            intercept = np.median(result)
            return intercept, slope
        else:
            raise ValueError('Invalid input sizes')
    else:
        raise ValueError('Invalid input dimensions')
