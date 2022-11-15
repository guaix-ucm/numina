#
# Copyright 2008-2022 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Background estimation

    Background estimation following Costa 1992, Bertin & Arnouts 1996
"""


import numpy
import scipy.ndimage as ndimage


def _interpolation(z, sx, sy, mx, my):
    # Spline to original size
    x, y = numpy.ogrid[-1:1:complex(0, mx), -1:1:complex(0, my)]
    newx, newy = numpy.mgrid[-1:1:complex(0, sx), -1:1:complex(0, sy)]

    x0 = x[0, 0]
    y0 = y[0, 0]
    dx = x[1, 0] - x0
    dy = y[0, 1] - y0
    ivals = (newx - x0)/dx
    jvals = (newy - y0)/dy
    coords = numpy.array([ivals, jvals])
    newf = ndimage.map_coordinates(z, coords)
    return newf


def background_estimator(bdata):
    """Estimate the background in a 2D array"""

    crowded = False

    std = numpy.std(bdata)
    std0 = std
    mean = bdata.mean()
    while True:
        prep = len(bdata)
        numpy.clip(bdata, mean - 3 * std, mean + 3 * std, out=bdata)
        if prep == len(bdata):
            if std < 0.8 * std0:
                crowded = True
            break
        std = numpy.std(bdata)
        mean = bdata.mean()

    if crowded:
        median = numpy.median(bdata)
        mean = bdata.mean()
        std = bdata.std()
        return 2.5 * median - 1.5 * mean, std

    return bdata.mean(), bdata.std()


def create_background_map(data, bsx, bsy):
    """Create a background map with a given mesh size"""
    sx, sy = data.shape
    mx = sx // bsx
    my = sy // bsy
    comp = []
    rms = []
    # Rows
    sp = numpy.split(data, numpy.arange(bsx, sx, bsx), axis=0)
    for s in sp:
        # Columns
        rp = numpy.split(s, numpy.arange(bsy, sy, bsy), axis=1)
        for r in rp:
            b, r = background_estimator(r)
            comp.append(b)
            rms.append(r)

    # Reconstructed image
    z = numpy.array(comp)
    z.shape = (mx, my)
    # median filter
    ndimage.median_filter(z, size=(3, 3), output=z)

    # Interpolate to the original size
    new = _interpolation(z, sx, sy, mx, my)

    # Interpolate the rms
    z = numpy.array(rms)
    z.shape = (mx, my)
    nrms = _interpolation(z, sx, sy, mx, my)

    return new, nrms
