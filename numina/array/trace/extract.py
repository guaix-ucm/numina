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

import numpy

from ._extract import extract_simple_intl
from .traces import axis_to_dispaxis


class Aperture(object):
    '''Spectroscopic aperture.'''
    def __init__(self, bbox, borders, axis=0, id=None):
        self.bbox = bbox
        self.borders = borders
        self.id = id
        self.axis = axis
        self.dispaxis = axis_to_dispaxis(axis)


def extract_simple_rss(arr, borders, axis=0, out=None):
    
    # If arr is not in native byte order, the C-extension won't work
    if arr.dtype.byteorder != '=':
        arr2 = arr.byteswap().newbyteorder()
    else:
        arr2 = arr
    
    if axis == 0:
        arr3 = arr2
    elif axis == 1:
        arr3 = arr2.T
    else:
        raise ValueError("'axis' must be 0 or 1")

    if out is None:
        out = numpy.zeros((len(borders), arr3.shape[1]), dtype='float')

    xx = numpy.arange(arr3.shape[1])

    # Borders contains a list of function objects
    for idx, (b1, b2) in enumerate(borders):
        bb1 = b1(xx)
        bb1[bb1 < -0.5] = -0.5    
        bb2 = b2(xx)
        bb2[bb2 > arr3.shape[0] - 0.5] = arr3.shape[0] - 0.5
        extract_simple_intl(arr3, xx, bb1, bb2, out[idx])
    return out


def extract_simple_rss_apers(arr, apers, axis=0, out=None):

    # If arr is not in native byte order, the C-extension won't work
    if arr.dtype.byteorder != '=':
        arr2 = arr.byteswap().newbyteorder()
    else:
        arr2 = arr

    if axis == 0:
        arr3 = arr2
        i1, i2 = 0, 2
    elif axis == 1:
        arr3 = arr2.T
        i1, i2 = 2, 4
    else:
        raise ValueError("'axis' must be 0 or 1")

    if out is None:
        out = numpy.zeros((len(apers), arr3.shape[1]), dtype='float')

    for idx, aper in enumerate(apers):
        if aper.axis != axis:
            raise ValueError("array 'axis' and aperture 'axis' are different")

        u1, u2 = aper.bbox[i1:i2]
        uu = numpy.arange(u1, u2 + 1)

        # Borders contains a list of function objects
        b1, b2 = aper.borders
        bb1 = b1(uu)
        bb1[bb1 < -0.5] = -0.5
        bb2 = b2(uu)
        bb2[bb2 > arr3.shape[0] - 0.5] = arr3.shape[0] - 0.5
        extract_simple_intl(arr3, uu, bb1, bb2, out[idx])

    return out
