#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Least squares 2D image fitting to a polynomial."""

import numpy


def _powers(order):
    for rank in range(order + 1):
        for bi in range(rank + 1):
            ai = rank - bi
            yield ai, bi


def _compute_value(power, wg):
    """Return the weight corresponding to single power."""
    if power not in wg:
        p1, p2 = power
        # y power
        if p1 == 0:
            yy = wg[(0, -1)]
            wg[power] = numpy.power(yy, p2 / 2).sum() / len(yy)
        # x power
        else:
            xx = wg[(-1, 0)]
            wg[power] = numpy.power(xx, p1 / 2).sum() / len(xx)
    return wg[power]


def _compute_weight(powers, wg):
    """Return the weight corresponding to given powers."""
    # split
    pow1 = (powers[0], 0)
    pow2 = (0, powers[1])

    cal1 = _compute_value(pow1, wg)
    cal2 = _compute_value(pow2, wg)

    return cal1 * cal2


def imsurfit(data, order, output_fit=False):
    """Fit a bidimensional polynomial to an image.

    :param data: a bidimensional array
    :param integer order: order of the polynomial
    :param bool output_fit: return the fitted image
    :returns: a tuple with an array with the coefficients of the polynomial terms

    >>> import numpy
    >>> xx, yy = numpy.mgrid[-1:1:100j,-1:1:100j]
    >>> z = 456.0 + 0.3 * xx - 0.9* yy
    >>> imsurfit(z, order=1) #doctest: +NORMALIZE_WHITESPACE
    (array([  4.56000000e+02,   3.00000000e-01,  -9.00000000e-01]),)

    """

    # we create a grid with the same number of points
    # between -1 and 1
    c0 = complex(0, data.shape[0])
    c1 = complex(0, data.shape[1])
    xx, yy = numpy.ogrid[-1:1:c0, -1:1:c1]

    ncoeff = (order + 1) * (order + 2) // 2

    powerlist = list(_powers(order))

    # Array with ncoff x,y moments of the data
    bb = numpy.zeros(ncoeff)
    # Moments
    for idx, powers in enumerate(powerlist):
        p1, p2 = powers
        bb[idx] = (data * xx ** p1 * yy ** p2).sum() / data.size

    # Now computing aa matrix
    # it contains \sum x^a y^b
    # most of the terms are zero
    # only those with a and b even remain
    # aa is symmetric

    x = xx[:, 0]
    y = yy[0]

    # weights are stored so we compute them only once
    wg = {(0, 0): 1, (-1, 0): x**2, (0, -1): y**2}
    # wg[(2,0)] = wg[(-1,0)].sum() / len(x)
    # wg[(0,2)] = wg[(0,-1)].sum() / len(y)

    aa = numpy.zeros((ncoeff, ncoeff))
    for j, ci in enumerate(powerlist):
        for i, ri in enumerate(powerlist[j:]):
            p1 = ci[0] + ri[0]
            p2 = ci[1] + ri[1]
            if p1 % 2 == 0 and p2 % 2 == 0:
                # val = (x ** p1).sum() / len(x) * (y ** p2).sum() / len(y)
                val = _compute_weight((p1, p2), wg)
                aa[j, i+j] = val

    # Making symmetric the array
    aa += numpy.triu(aa, k=1).T

    polycoeff = numpy.linalg.solve(aa, bb)

    if output_fit:
        index = 0
        result = 0
        for o in range(order + 1):
            for b in range(o + 1):
                a = o - b
                result += polycoeff[index] * (xx ** a) * (yy ** b)
                index += 1

        return polycoeff, result

    return (polycoeff,)


class FitOne(object):
    def __init__(self, x, y, z):
        """Fit a plane to a region using least squares."""
        x = x.astype('float')
        y = y.astype('float')
        self.xm = x.mean()
        self.ym = y.mean()
        x -= self.xm
        y -= self.ym

        aa = numpy.zeros(shape=(3, 3))
        aa[0, 0] = 1
        aa[1, 1] = (x*x).mean()
        aa[2, 2] = (y*y).mean()
        aa[1, 2] = aa[2, 1] = (x*y).mean()
        bb = [z.mean(), (x*z).mean(), (y*z).mean()]

        self.pf = numpy.linalg.solve(aa, bb)

    def __call__(self, x, y):
        xf = x.astype('float') - self.xm
        yf = y.astype('float') - self.ym
        return self.pf[0] + self.pf[1] * xf + self.pf[2] * yf
