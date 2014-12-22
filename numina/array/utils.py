#
# Copyright 2013-2014 Universidad Complutense de Madrid
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

'''Utility routines'''

from __future__ import division

import math

import numpy


def wc_to_pix_1d(w):
    '''Return the pixel where a value is located.'''
    return int(math.floor(w + 0.5))


def wcs_to_pix(w):
    return [wc_to_pix_1d(w1) for w1 in w[::-1]]


def wcs_to_pix_np(w):
    wnp = numpy.asarray(w)
    mm = numpy.floor(wnp + 0.5)
    return mm[::-1].astype('int')


def slice_create(center, block, start=0, stop=None):
    '''Return an slice with a symmetric region around center.'''

    do = wc_to_pix_1d(center - block)
    up = wc_to_pix_1d(center + block)

    l = max(start, do)

    if stop is not None:
        h = min(up + 1, stop)
    else:
        h = up + 1

    return slice(l, h, 1)


def image_box(center, shape, box):
    '''Create a region of size box, around a center in a image of shape.'''
    return tuple(slice_create(c, b, stop=s)
                 for c, s, b in zip(center, shape, box))


def expand_slice(s, a, b, start=0, stop=None):
    '''Expand a slice on the start/stop limits'''
    n1 = max(s.start - a, start)
    n2 = s.stop + b
    if stop is not None:
        n2 = min(n2, stop)

    return slice(n1, n2, 1)


def expand_region(tuple_of_s, a, b, start=0, stop=None):
    '''Apply expend_slice on a tuple of slices'''
    return tuple(expand_slice(s, a, b, start=start, stop=stop)
                 for s in tuple_of_s)
