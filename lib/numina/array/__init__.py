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

import numpy
import pyfits
from scipy import asarray, zeros_like, minimum, maximum

from numina.array.combine import median

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
    # Computing final image size and new offsets
    sharr = numpy.asarray(shapes)

    offarr = numpy.asarray(offsets)        
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)     
    finalshape = ucorners.max(axis=0) - ref 
    offsetsp = offarr - ref
    return (finalshape, offsetsp)

def resize_array(data, finalshape, region):
    newdata = numpy.zeros(finalshape, dtype=data.dtype)
    newdata[region] = data
    return newdata

def flatcombine(data, masks, scales=None, blank=1.0):
    # Zero masks
    # TODO Do a better fix here
    # This is to avoid negative of zero values in the flat field
    #if scales is not None:
    #    maxscale = max(scales)
    #    scales = [s / maxscale for s in scales]
    
    sf_data = median(data, masks, scales=scales)
    
    mm = sf_data[0] <= 0
    sf_data[0][mm] = blank
    return sf_data

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

