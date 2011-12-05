#
# Copyright 2008-2011 Universidad Complutense de Madrid
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
import logging
from itertools import imap, product

import numpy # pylint: disable-msgs=E1101
from scipy import asarray, zeros_like, minimum, maximum
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

from numina.array.imsurfit import FitOne

_logger = logging.getLogger("numina.array")

def subarray_match(shape, ref, sshape, sref=None):
    '''Compute the slice representation of intersection of two arrays.
    
    Given the shapes of two arrays and a reference point ref, compute the
    intersection of the two arrays.
    It returns a tuple of slices, that can be passed to the two images as indexes
    
    :param shape: the shape of the reference array
    :param ref: coordinates of the reference point in the first array system
    :param sshape: the shape of the second array
    :param: sref: coordinates of the reference point in the second array system, the origin by default
    :type sref: sequence or None
    :return: two matching slices, corresponding to both arrays or a tuple of Nones if they don't match
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

def resize_array(data, finalshape, region, fill=0):
    newdata = numpy.empty(finalshape, dtype=data.dtype)
    newdata.fill(fill)
    newdata[region] = data
    return newdata

def fixpix(data, mask, kind='linear'):
    '''Interpolate 2D array data in rows'''
    if data.shape != mask.shape:
        raise ValueError

    if not numpy.any(mask):
        return data

    x = numpy.arange(0, data.shape[0])    
    for row, mrow in zip(data, mask):
        if numpy.any(mrow): # Interpolate if there's some pixel missing
            itp = interp1d(x[mrow == False], row[mrow == False], kind=kind, copy=False)
            invalid = mrow == True
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
        dilmask =  numpy.logical_or(back, segm)
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
        calc = FitOne(xi, yi, out[xi,yi])
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

def blockgen1d(block, size):
    '''Compute 1d block intervals to be used by combine.
    
    blockgen1d computes the slices by recursively halving the initial
    interval (0, size) by 2 until its size is lesser or equal than block
    
    :param block: an integer maximum block size
    :param size: original size of the interval, it corresponds to a 0:size slice
    :return: a list of slices
    
    Example:
    
        >>> blockgen1d(512, 1024)
        [slice(0, 512, None), slice(512, 1024, None)]
        
        >>> blockgen1d(500, 1024)
        [slice(0, 256, None), slice(256, 512, None), slice(512, 768, None), slice(768, 1024, None)]
    
    '''
    def numblock(block, x):
        '''Compute recursively the numeric intervals
        '''
        a, b = x
        if b - a <= block:
            return [x]
        else:
            result = []
            d = int(b - a) / 2
            for i in imap(numblock, [block, block], [(a, a + d), (a + d, b)]):
                result.extend(i)
            return result
        
    return [slice(*l) for l in numblock(block, (0, size))]


def blockgen(blocks, shape):
    '''Generate a list of slice tuples to be used by combine.
    
    The tuples represent regions in an N-dimensional image.
    
    :param blocks: a tuple of block sizes
    :param shape: the shape of the n-dimensional array
    :return: an iterator to the list of tuples of slices
    
    Example:
        
        >>> blocks = (500, 512)
        >>> shape = (1040, 1024)
        >>> for i in blockgen(blocks, shape):
        ...     print i
        (slice(0, 260, None), slice(0, 512, None))
        (slice(0, 260, None), slice(512, 1024, None))
        (slice(260, 520, None), slice(0, 512, None))
        (slice(260, 520, None), slice(512, 1024, None))
        (slice(520, 780, None), slice(0, 512, None))
        (slice(520, 780, None), slice(512, 1024, None))
        (slice(780, 1040, None), slice(0, 512, None))
        (slice(780, 1040, None), slice(512, 1024, None))
        
    
    '''
    iterables = [blockgen1d(l, s) for (l, s) in zip(blocks, shape)]
    return product(*iterables)


if __name__ == '__main__':
    import _ufunc
    from timeit import Timer
    
    
    a = numpy.arange(10000)
    
    def test1():
        _ufunc.test3(a)
    
    def test2():
        numpy.median(a)

    t = Timer("test1()", "from __main__ import test1")
    print t.timeit()
    
    t = Timer("test2()", "from __main__ import test2")
    print t.timeit()

