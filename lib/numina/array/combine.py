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

from itertools import izip, imap, product

import numpy as np

from numina.array._combine import internal_combine, internal_combine_with_offsets
from numina.array._combine import CombineError


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
   
def combine_shape(shapes, offsets):
    # Computing final image size and new offsets
    sharr = np.asarray(shapes)
    
    offarr = np.asarray(offsets)        
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)        
    finalshape = ucorners.max(axis=0) - ref 
    offsetsp = offarr - ref
    return (finalshape, offsetsp)

def _combine(method, images, masks=None, dtype=None, out=None,
             args=(), zeros=None, scales=None, weights=None, offsets=None,):
        # method should be a string
    if not isinstance(method, basestring) and not callable(method):
        raise CombineError('method is neither a string or callable')
    
    # Check inputs
    if not images:
        raise CombineError("len(inputs) == 0")

    number_of_images = len(images)
    images = map(np.asanyarray, images)
    
    # All images have the same shape
    allshapes = [i.shape for i in images]
    baseshape = images[0].shape
    if any(shape != baseshape for shape in allshapes[1:]):
        raise CombineError("Images don't have the same shape")
    
    # Offsets
    if offsets is None:
        finalshape = baseshape
    else:
        if len(images) != len(offsets):
            raise CombineError("len(inputs) != len(offsets)")
        
        finalshape, offsets = combine_shape(allshapes, offsets)
        offsets = offsets.astype('int')
        
    if masks:
        if len(images) != len(masks):
            raise CombineError("len(inputs) != len(masks)")
    
        # Error if mask and image have different shape
        if any(imshape != ma.shape for (imshape, ma) in izip(allshapes, masks)):
            raise CombineError("mask and image have different shape")
    else:
        masks = [np.zeros(baseshape, dtype=np.bool)] * number_of_images
        
    # Creating out if needed
    # We need three numbers
    outshape = (3,) + tuple(finalshape)
    
    if out is None:
        out = np.zeros(outshape, dtype='float')
    else:
        if out.shape != outshape:
            raise CombineError("result has wrong shape")  
    
    if zeros is None:
        zeros = np.zeros(number_of_images, dtype='float')
    else:
        zeros = np.asanyarray(zeros, dtype='float')
        if zeros.shape != (number_of_images,):
            raise CombineError('incorrect number of zeros')
        
    if scales is None:
        scales = np.ones(number_of_images, dtype='float')
    else:
        scales = np.asanyarray(scales, dtype='float')
        if scales.shape != (number_of_images,):
            raise CombineError('incorrect number of scales')
        
    if weights is None:
        weights = np.ones(number_of_images, dtype='float')
    else:
        weights = np.asanyarray(scales, dtype='float')
        if weights.shape != (number_of_images,):
            raise CombineError('incorrect number of weights')

    if offsets is None:
        internal_combine(method, images, masks, out0=out[0], out1=out[1], out2=out[2], args=args, 
                         zeros=zeros, scales=scales, weights=weights)
    else:
        internal_combine_with_offsets(method, images, masks, out0=out[0], out1=out[1], out2=out[2], 
                                      args=args, zeros=zeros, scales=scales, weights=weights, 
                                      offsets=offsets)
    
    return out.astype(dtype)

def mean(images, masks=None, dtype=None, out=None, zeros=None, scales=None,
         weights=None, dof=0, offsets=None):
    '''Combine images using the mean, with masks and offsets.
    
    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.
    
    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.
    
    :param images: a list of arrays
    :param masks: a list of masked arrays, True values are masked
    :param dtype: data type of the ouput
    :param out: optional output, with one more axis than the input images
    :param dof: degrees of freedom 
    :return: mean, variance and number of points stored in
    :raise TypeError: if method is not callable
    :raise CombineError: if method is not callable
    
    
    
    Example:
    
       >>> import numpy
       >>> image = numpy.array([[1.,3.],[1., -1.4]])
       >>> inputs = [image, image + 1]
       >>> mean(inputs) #doctest: +NORMALIZE_WHITESPACE
       array([[[ 1.5 ,  3.5 ],
               [ 1.5 , -0.9 ]],
              [[ 0.25,  0.25],
               [ 0.25,  0.25]],
              [[ 2.  ,  2.  ],
               [ 2.  ,  2.  ]]])
       
    '''
    return _combine('mean', images, masks=masks, dtype=dtype, out=out, args=(dof,),
                    zeros=zeros, scales=scales, weights=weights, offsets=offsets)
    
    
    
def median(images, masks=None, dtype=None, out=None, zeros=None, scales=None, 
           weights=None, offsets=None):
    '''Combine images using the median, with masks.
    
    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.
    
    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.
    
    :param images: a list of arrays
    :param masks: a list of masked arrays, True values are masked
    :param dtype: data type of the ouput
    :param out: optional output, with one more axis than the input images
 
    :return: mean, variance and number of points stored in
    :raise TypeError: if method is not callable
    :raise CombineError: if method is not callable
       
    '''
    return _combine('median', images, masks=masks, dtype=dtype, out=out,
                    zeros=zeros, scales=scales, weights=weights, offsets=offsets)    

def sigmaclip(images, masks=None, dtype=None, out=None, zeros=None, scales=None,
         weights=None, offsets=None, low=4., high=4., dof=0):
    '''Combine images using the sigma-clipping, with masks.
    
    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.
    
    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.
    
    :param images: a list of arrays
    :param masks: a list of masked arrays, True values are masked
    :param dtype: data type of the ouput
    :param out: optional output, with one more axis than the input images
    :param low:
    :param high:
    :param dof: degrees of freedom 
    :return: mean, variance and number of points stored in    
    '''
    
    return _combine('sigmaclip', images, masks=masks, dtype=dtype, out=out,
                    args=(low, high, dof), zeros=zeros, scales=scales, 
                    weights=weights, offsets=offsets)


if __name__ == "__main__":
    from numina.decorators import print_timing
      
    tmean = print_timing(mean)
    
    # Inputs
    shape = (2048, 2048)
    data_dtype = 'int16'
    nimages = 10
    minputs = [i * np.ones(shape, dtype=data_dtype) for i in xrange(nimages)]
    mmasks = [np.zeros(shape, dtype='int16') for i in xrange(nimages)]
    offsets = np.array([[0, 0]] * nimages, dtype='int16') 
    
    print 'Computing'
    for i in range(1):
        outrr = tmean(minputs, mmasks, offsets=offsets)
        print outrr[2]
        outrr = tmean(minputs, mmasks)
        print outrr[2]
