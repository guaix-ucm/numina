#
# Copyright 2008-2010 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 
import tempfile
import subprocess
import logging
from itertools import imap, product

import numpy
import pyfits
from scipy import asarray, zeros_like, minimum, maximum

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

def resize_array(data, finalshape, region):
    newdata = numpy.zeros(finalshape, dtype=data.dtype)
    newdata[region] = data
    return newdata

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
    d = img.data[region]
    m = omask.data[region]
    median_sky = numpy.median(d[m == 0])
    return median_sky

def create_object_mask(array, segmask_name=None):

    if segmask_name is None:
        ck_img = tempfile.NamedTemporaryFile(prefix='emir_', dir='.')
    else:
        ck_img = segmask_name

    # A temporary filename used to store the array in fits format
    tf = tempfile.NamedTemporaryFile(prefix='emir_', dir='.')
    pyfits.writeto(filename=tf, data=array)
    
    # Run sextractor, it will create a image called check.fits
    # With the segmentation _masks inside
    sub = subprocess.Popen(["sex",
                            "-CHECKIMAGE_TYPE", "SEGMENTATION",
                            "-CHECKIMAGE_NAME", ck_img,
                            '-VERBOSE_TYPE', 'QUIET',
                            tf.name],
                            stdout=subprocess.PIPE)
    sub.communicate()

    # Read the segmentation image
    result = pyfits.getdata(ck_img)

    # Close the tempfile
    tf.close()
    
    if segmask_name is None:
        ck_img.close()

    return result

def numberarray(x, shape=(5, 5)):
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

