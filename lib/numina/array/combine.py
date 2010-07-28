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

from numina.array import combine_shape
from numina.array._combine import internal_combine, internal_combine_with_offsets
from numina.array._combine import CombineError

COMBINE_METHODS = {'average': [('dof', 1)], 'median': []}
REJECT_METHODS = {'none': [], 'sigmaclip': [('low', 4.), ('high', 4.)], 
                  'minmax': [('nlow', 0), ('nhigh', 0)]}

def merge_default(sequence, defaults):
    '''Return *sequence* elements, with *defaults* as default values.
    
    The length of the result sequence is equal to the length
    of *defaults*. If values are missing from *sequence*, they are taken from
    the same position in the sequence *defaults*.
    
    >>> list(merge_default([-4, 4, 8], defaults=[]))
    []
    >>> list(merge_default([-4, 4], defaults=[10, 20, -10, 0]))
    [-4, 4, -10, 0]
    
    '''
    iter1 = iter(sequence)
    iter2 = iter(defaults)
    
    while True:
        dfc = next(iter2)
        try:
            val = next(iter1)
            yield val
        except StopIteration:
            yield dfc
            
            while True:
                yield next(iter2)

def assign_def_values(args, pars, name=None):
    '''Assign default values from *pars* to *args*.
    
    *pars* is a list with tuples of pairs containing
    the parameter name and its value.
    
    >>> assign_def_values((-10, 20), [('par1', 10), ('par2', 30)])
    (-10, 20)
    >>> assign_def_values((-10,), [('par1', 10), ('par2', 30)])
    (-10, 30)
    
    Raises ValueError if *args* is longer than *pars*
    
    '''
    if not args and pars:
        # default arguments for method
        # zip(*) is the equivalent to unzip
        _, args = zip(*pars)
    else:
        npars = len(pars)
        nargs = len(args)
        if npars < nargs:
            # We are passing more parameters than needed
            name = name or 'function'
            raise ValueError('%s is receiving %d parameters, only %d are needed' % 
                               (name, nargs, npars))
        elif npars > nargs:
            args = merge_default(args, map(lambda x:x[1], pars))
    return tuple(args)

def combine(images, masks=None, dtype=None, out=None,
            method='average', margs=(), reject='none', rargs=(), 
            zeros=None, scales=None, weights=None, offsets=None):
        
    WORKTYPE = 'float'
    
    if method not in COMBINE_METHODS:
        raise CombineError('method is not an allowed string: %s' % COMBINE_METHODS)
    
    if reject not in REJECT_METHODS:
        raise CombineError('rejection method is not an allowed string: %s' % REJECT_METHODS)
    
    try:
        margs = assign_def_values(margs, COMBINE_METHODS[method], method)
        rargs = assign_def_values(rargs, REJECT_METHODS[reject], reject)
    except ValueError, err:
        raise CombineError('%s' % err)
        
    # Check inumpy.uts
    if not images:
        raise CombineError("len(inputs) == 0")

    number_of_images = len(images)
    
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
                   method='average', margs=(dof,), reject='none', 
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
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input images
 
    :return: mean, variance and number of points stored in
    :raise TypeError: if method is not callable
    :raise CombineError: if method is not callable
       
    '''
    return combine(images, masks=masks, dtype=dtype, out=out,
                   method='median', reject='none',
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
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input images
    :param low:
    :param high:
    :param dof: degrees of freedom 
    :return: mean, variance and number of points stored in    
    '''
    
    return combine(images, masks=masks, dtype=dtype, out=out,
                   method='average', margs=(dof,), 
                   reject='sigmaclip', rargs=(low, high), 
                   zeros=zeros, scales=scales, weights=weights, offsets=offsets)

def flatcombine(data, masks, dtype=None, scales=None, 
                blank=1.0, method='median', margs=()):
    
    result = combine(data, masks=masks, 
                     dtype=dtype, scales=scales, 
                     method=method, margs=margs)
    
    # Sustitute values <= 0 by blank
    mm = result[0] <= 0
    result[0][mm] = blank
    return result

def zerocombine(data, masks, dtype=None, scales=None, 
                method='median', margs=()):
    
    result = combine(data, masks=masks, 
                     dtype=dtype, scales=scales, 
                     method=method, margs=margs)

    return result
