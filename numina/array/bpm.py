#
# Copyright 2016 Universidad Complutense de Madrid
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

"""Fix points in an image given by a bad pixel mask """

from __future__ import division


import numpy

from numina.array._bpm import _process_bpm_intl


def process_bpm(method, arr, mask, hwin=2, wwin=2, fill=0):

    # FIXME: we are not considering variance extension
    # If arr is not in native byte order, the cython extension won't work
    if arr.dtype.byteorder != '=':
        narr = arr.byteswap().newbyteorder()
    else:
        narr = arr
    out = numpy.empty_like(narr, dtype='double')

    # Casting, Cython doesn't support well type bool
    cmask = numpy.where(mask > 0, 1, 0).astype('uint8')

    _process_bpm_intl(method, narr, cmask, out, hwin=hwin, wwin=wwin, fill=fill)
    return out


def process_bpm_median(arr, mask, hwin=2, wwin=2, fill=0):
    import numina.array.combine

    method = numina.array.combine.median_method()

    return process_bpm(method, arr, mask, hwin=hwin, wwin=wwin, fill=fill)
