#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

from __future__ import print_function

import logging
from itertools import imap, product

import numpy
from scipy import asarray, zeros_like, minimum, maximum
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

from .blocks import blockgen1d, blockgen
from .imsurfit import FitOne
from numina.array.nirproc import ramp_array
from numina.array.nirproc import fowler_array

_logger = logging.getLogger("numina.array")


def subarray_match(shape, ref, sshape, sref=None):
    '''Compute the slice representation of intersection of two arrays.

    Given the shapes of two arrays and a reference point ref, compute the
    intersection of the two arrays.
    It returns a tuple of slices, that can be passed to the two \
    images as indexes

    :param shape: the shape of the reference array
    :param ref: coordinates of the reference point in the first array system
    :param sshape: the shape of the second array
    :param: sref: coordinates of the reference point in the \
    second array system, the origin by default
    :type sref: sequence or None
    :return: two matching slices, corresponding to both arrays \
    or a tuple of Nones if they don't match
    :rtype: a tuple

    Example:

      >>> import numpy
      >>> im = numpy.zeros((1000, 1000))
      >>> sim = numpy.ones((40, 40))
      >>> i,j = subarray_match(im.shape, [20, 23], sim.shape)
      >>> im[i] = 2 * sim[j]

    '''
    # Reference point in im
    ref1 = asarray(ref, dtype='int')

    if sref is not None:
        ref2 = asarray(sref, dtype='int')
    else:
        ref2 = zeros_like(ref1)

    offset = ref1 - ref2
    urc1 = minimum(offset + asarray(sshape) - 1, asarray(shape) - 1)
    blc1 = maximum(offset, 0)
    urc2 = urc1 - offset
    blc2 = blc1 - offset

    def valid_slice(b, u):
        if b >= u + 1:
            return None
        else:
            return slice(b, u + 1)

    f = tuple(valid_slice(b, u) for b, u in zip(blc1, urc1))
    s = tuple(valid_slice(b, u) for b, u in zip(blc2, urc2))

    if not all(f) or not all(s):
        return (None, None)

    return (f, s)


def combine_shape(shapes, offsets):
    # Computing final array size and new offsets
    sharr = asarray(shapes)
    offarr = asarray(offsets)
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)
    finalshape = ucorners.max(axis=0) - ref
    offsetsp = offarr - ref
    return (finalshape, offsetsp)


def resize_array(data, finalshape, region, window=None,
                 scale=1, fill=0, conserve=True):

    if window:
        data = data[window]

    if scale == 1:
        finaldata = data
    else:
        finaldata = rebin_scale(data, scale)

    newdata = numpy.empty(finalshape, dtype=data.dtype)
    newdata.fill(fill)
    newdata[region] = finaldata
    # Conserve the total sum of the original data
    if conserve:
        newdata[region] /= scale**2
    return newdata


def rebin_scale(a, scale=1):
    '''Scale an array to a new shape.'''

    newshape = tuple((side * scale) for side in a.shape)

    slices = [slice(0, old, float(old)/new)
              for old, new in zip(a.shape, newshape)]
    coordinates = numpy.mgrid[slices]
    # choose the biggest smaller integer index
    indices = coordinates.astype('i')
    return a[tuple(indices)]


def rebin(a, newshape):
    '''Rebin an array to a new shape.'''

    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old)/new)
              for old, new in zip(a.shape, newshape)]
    coordinates = numpy.mgrid[slices]
    # choose the biggest smaller integer index
    indices = coordinates.astype('i')
    return a[tuple(indices)]


def fixpix(data, mask, kind='linear'):
    '''Interpolate 2D array data in rows'''
    if data.shape != mask.shape:
        raise ValueError

    if not numpy.any(mask):
        return data

    x = numpy.arange(0, data.shape[0])
    for row, mrow in zip(data, mask):
        if numpy.any(mrow):  # Interpolate if there's some pixel missing
            valid = (mrow == numpy.False_)
            invalid = (mrow == numpy.True_)
            itp = interp1d(x[valid], row[valid], kind=kind, copy=False)
            row[invalid] = itp(x[invalid]).astype(row.dtype)
    return data


def fixpix2(data, mask, iterations=3, out=None):
    '''Substitute pixels in mask by a bilinear least square fitting.
    '''
    out = out if out is not None else data.copy()

    # A binary mask, regions are ones
    binry = mask != 0

    # Label regions in the binary mask
    lblarr, labl = ndimage.label(binry)

    # Structure for dilation is 8-way
    stct = ndimage.generate_binary_structure(2, 2)
    # Pixels in the background
    back = lblarr == 0
    # For each object
    for idx in range(1, labl + 1):
        # Pixels of the object
        segm = lblarr == idx
        # Pixels of the object or the background
        # dilation will only touch these pixels
        dilmask = numpy.logical_or(back, segm)
        # Dilation 3 times
        more = ndimage.binary_dilation(segm, stct,
                                       iterations=iterations,
                                       mask=dilmask)
        # Border pixels
        # Pixels in the border around the object are
        # more and (not segm)
        border = numpy.logical_and(more, numpy.logical_not(segm))
        # Pixels in the border
        xi, yi = border.nonzero()
        # Bilinear leastsq calculator
        calc = FitOne(xi, yi, out[xi, yi])
        # Pixels in the region
        xi, yi = segm.nonzero()
        # Value is obtained from the fit
        out[segm] = calc(xi, yi)

    return out


def correct_dark(data, dark, dtype='float32'):
    result = data - dark
    result = result.astype(dtype)
    return result


def correct_flatfield(data, flat, dtype='float32'):
    result = data / flat
    result = result.astype(dtype)
    return result


def correct_sky(data, sky, dtype='float32'):
    result = data - sky
    result = result.astype(dtype)
    return result


def correct_nonlinearity(data, polynomial, dtype='float32'):
    result = numpy.polyval(polynomial, data)
    result = result.astype(dtype)
    return result


def compute_sky_advanced(data, omasks):
    from numina.array.combine import median
    d = data[0]
    m = omasks[0]
    median_sky = numpy.median(d[m == 0])
    result = numpy.zeros(data[0].shape)
    result += median_sky
    return result

    result = median(data, omasks)
    return result[0]


def compute_median_background(img, omask, region):
    d = img[region]
    m = omask[region]
    median_sky = numpy.median(d[m == 0])
    return median_sky


def numberarray(x, shape):
    '''Return x if it is an array or create an array and fill it with x.'''
    try:
        iter(x)
    except TypeError:
        return numpy.ones(shape) * x
    else:
        return x
