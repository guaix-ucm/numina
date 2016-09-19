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

"""Robust fits"""

import numpy


def fit_theil_sen(x, y):
    """Compute a robust linear fit using the Theil-Sen method.

    See http://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator for details.
    This function "pairs up sample points by the rank of their x-coordinates
    (the point with the smallest coordinate being paired with the first point
    above the median coordinate, etc.) and computes the median of the slopes of
    the lines determined by these pairs of points".

    Parameters
    ----------
    x : array_like, shape (M,)
        X coordinate array.
    y : array_like, shape (M,) or (M,K)
        Y coordinate array. If the array is two dimensional, each column of
        the array is independently fitted sharing the same x-coordinates. In
        this last case, the returned intercepts and slopes are also 1d numpy
        arrays.

    Returns
    -------
    coef : ndarray, shape (2,) or (2, K)
           Intercept and slope of the linear fit. If y was 2-D, the
           coefficients in column k of coef represent the linear fit
           to the data in y's k-th column.

    Raises
    ------
    ValueError:
        If the number of points to fit is < 5

    """

    xx = numpy.asarray(x)
    y1 = numpy.asarray(y)
    n = len(xx)
    if n < 5:
        raise ValueError('Number of points < 5')

    if xx.ndim != 1:
        raise ValueError('Input arrays have unexpected dimensions')

    if y1.ndim == 1:
        if len(y1) != n:
            raise ValueError('X and Y arrays have different sizes')
        yy = y1[numpy.newaxis, :]
    elif y1.ndim == 2:
        if n != y1.shape[0]:
            raise ValueError(
                'Y-array size in the fitting direction is different to the X-array size')
        yy = y1.T
    else:
        raise ValueError('Input arrays have unexpected dimensions')

    nmed = n // 2
    iextra = nmed if (n % 2) == 0 else nmed + 1

    deltx = xx[iextra:] - xx[:nmed]
    delty = yy[:, iextra:] - yy[:, :nmed]
    allslopes = delty / deltx
    slopes = numpy.median(allslopes, axis=1)
    allinters = yy - slopes[:, numpy.newaxis] * x
    inters = numpy.median(allinters, axis=1)

    coeff = numpy.array([inters, slopes])
    return numpy.squeeze(coeff)
