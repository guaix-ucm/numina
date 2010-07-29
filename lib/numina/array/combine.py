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

'''Different methods for combining lists of arrays.'''

from itertools import izip

import numpy

from numina.recipes.registry import Schema
from numina.array import combine_shape
from numina.array._combine import internal_combine, internal_combine_with_offsets
from numina.array._combine import CombineError

COMBINE_METHODS = {'average': [Schema('dof', 1, 'Description')], 
                   'median': []
                   }

REJECT_METHODS = {'none': [], 
                  'sigmaclip': [Schema('low', 4., 'Description'), 
                                Schema('high', 4., 'Description')], 
                  'minmax': [Schema('nlow', 0, 'Description'), 
                             Schema('nhigh', 0, 'Description')]
                  }

def combine(images, masks=None, dtype=None, out=None,
            method='average', reject='none', 
            zeros=None, scales=None, weights=None, 
            offsets=None, **kwds):
    '''Stack arrays using different methods.'''
        
    WORKTYPE = 'float'
    
    if method not in COMBINE_METHODS:
        raise CombineError('method is not an allowed string: %s' % COMBINE_METHODS)
    
    if reject not in REJECT_METHODS:
        raise CombineError('rejection method is not an allowed string: %s' % REJECT_METHODS)
        
    margs = tuple(kwds.get(par, dft) for par, dft, _ in COMBINE_METHODS[method])
    rargs = tuple(kwds.get(par, dft) for par, dft, _ in REJECT_METHODS[reject])
        
    # Check inumpy.uts
    if not images:
        raise CombineError("len(inputs) == 0")

    number_of_images = len(images)
    # checking the parameters of the different methods
    if reject == 'minmax':
        if rargs[0] + rargs[1] > number_of_images:
            raise CombineError('minmax rejection and rejected points > number of images')
    
    images = map(numpy.asanyarray, images)
    
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
        masks = [numpy.zeros(baseshape, dtype='bool')] * number_of_images
        
    # Creating out if needed
    # We need three numbers
    outshape = (3,) + tuple(finalshape)
    
    if out is None:
        out = numpy.zeros(outshape, dtype=WORKTYPE)
    else:
        if out.shape != outshape:
            raise CombineError("result has wrong shape")  
    
    if zeros is None:
        zeros = numpy.zeros(number_of_images, dtype=WORKTYPE)
    else:
        zeros = numpy.asanyarray(zeros, dtype=WORKTYPE)
        if zeros.shape != (number_of_images,):
            raise CombineError('incorrect number of zeros')
        
    if scales is None:
        scales = numpy.ones(number_of_images, dtype=WORKTYPE)
    else:
        scales = numpy.asanyarray(scales, dtype=WORKTYPE)
        if scales.shape != (number_of_images,):
            raise CombineError('incorrect number of scales')
        
    if weights is None:
        weights = numpy.ones(number_of_images, dtype=WORKTYPE)
    else:
        weights = numpy.asanyarray(scales, dtype=WORKTYPE)
        if weights.shape != (number_of_images,):
            raise CombineError('incorrect number of weights')

    if offsets is None:
        internal_combine(images, masks, out0=out[0], out1=out[1], out2=out[2], 
                         method=method, margs=margs, reject=reject, rargs=rargs, 
                         zeros=zeros, scales=scales, weights=weights)
    else:
        internal_combine_with_offsets(images, masks, 
                                      out0=out[0], out1=out[1], out2=out[2], offsets=offsets,
                                      method=method, margs=margs, reject=reject, rargs=rargs, 
                                      zeros=zeros, scales=scales, weights=weights, 
                                      )
    
    return out.astype(dtype)


def mean(images, masks=None, dtype=None, out=None,
         reject='none', zeros=None, scales=None,
         weights=None, offsets=None, dof=0, **kwds):
    '''Combine images using the mean, with masks and offsets.
    
    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.
    
    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.
    
    :param images: a list of arrays
    :param masks: a list of masked arrays, True values are masked
    :param dtype: data type of the output
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
    return combine(images, masks=masks, dtype=dtype, out=out, 
                   method='average', reject=reject, 
                   zeros=zeros, scales=scales, weights=weights, 
                   offsets=offsets, dof=dof, **kwds)
    
    
    
def median(images, masks=None, dtype=None, out=None, 
           reject='none', zeros=None, scales=None, 
           weights=None, offsets=None, **kwds):
    '''Combine images using the median, with masks.
    
    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.
    
    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.
    
    :param images: a list of arrays
    :param masks: a list of masked arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input images
 
    :return: mean, variance and number of points stored in
    :raise TypeError: if method is not callable
    :raise CombineError: if method is not callable
       
    '''
    return combine(images, masks=masks, dtype=dtype, out=out,
                   method='median', reject=reject,
                   zeros=zeros, scales=scales, weights=weights, 
                   offsets=offsets, **kwds)    

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
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input images
    :param low:
    :param high:
    :param dof: degrees of freedom 
    :return: mean, variance and number of points stored in    
    '''
    
    return combine(images, masks=masks, dtype=dtype, out=out,
                   method='average', reject='sigmaclip', 
                   zeros=zeros, scales=scales, weights=weights, 
                   offsets=offsets, dof=dof, high=high, low=low)

def flatcombine(data, masks, dtype=None, scales=None, 
                blank=1.0, method='median', reject='none', **kwds):
    
    result = combine(data, masks=masks, 
                     dtype=dtype, scales=scales, 
                     method=method, reject=reject, **kwds)
    
    # Sustitute values <= 0 by blank
    mm = result[0] <= 0
    result[0][mm] = blank
    return result

def zerocombine(data, masks, dtype=None, scales=None, 
                method='median', reject='none', **kwds):
    
    result = combine(data, masks=masks, 
                     dtype=dtype, scales=scales, 
                     method=method, reject=reject, **kwds)

    return result

if __name__ == '__main__':
    print combine([numpy.ones((100, 100))] * 3, method='average', reject='minmax')

