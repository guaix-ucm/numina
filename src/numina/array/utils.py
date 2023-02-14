#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Utility routines"""

import math

import numpy


def coor_to_pix_1d(w):
    """Return the pixel where a coordinate is located."""
    return int(math.floor(w + 0.5))


def wcs_to_pix(w):
    """

    Parameters
    ----------
    w

    Returns
    -------

    """
    return [coor_to_pix_1d(w1) for w1 in w[::-1]]


def coor_to_pix(w, order='rc'):
    """

    Parameters
    ----------
    w
    order

    Returns
    -------

    """
    if order == 'xy':
        return coor_to_pix(w[::-1], order='rc')
    return [coor_to_pix_1d(w1) for w1 in w]


def wcs_to_pix_np(w):
    """

    Parameters
    ----------
    w

    Returns
    -------

    """
    wnp = numpy.asarray(w)
    mm = numpy.floor(wnp + 0.5)
    return mm[::-1].astype('int')


def slice_create(center, block, start=0, stop=None):
    """
    Return an slice with a symmetric region around center.

    Parameters
    ----------
    center
    block
    start
    stop

    Returns
    -------

    """

    do = coor_to_pix_1d(center - block)
    up = coor_to_pix_1d(center + block)

    l = max(start, do)

    if stop is not None:
        h = min(up + 1, stop)
    else:
        h = up + 1

    return slice(l, h, 1)


def image_box(center, shape, box):
    """Create a region of size box, around a center in a image of shape."""
    return tuple(slice_create(c, b, stop=s)
                 for c, s, b in zip(center, shape, box))


def image_box2d(x, y, shape, box):
    """

    Parameters
    ----------
    x
    y
    shape
    box

    Returns
    -------

    """
    return image_box((y, x), shape, box)


def extent(sl):
    """

    Parameters
    ----------
    sl

    Returns
    -------

    """
    result = [sl[1].start-0.5, sl[1].stop-0.5, sl[0].start-0.5, sl[0].stop-0.5]
    return result


def expand_slice(s, a, b, start=0, stop=None):
    """Expand a slice on the start/stop limits"""
    n1 = max(s.start - a, start)
    n2 = s.stop + b
    if stop is not None:
        n2 = min(n2, stop)

    return slice(n1, n2, 1)


def expand_region(tuple_of_s, a, b, start=0, stop=None):
    """Apply expend_slice on a tuple of slices"""
    return tuple(expand_slice(s, a, b, start=start, stop=stop)
                 for s in tuple_of_s)
