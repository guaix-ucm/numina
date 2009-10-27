#
# Copyright 2008-2009 Sergio Pascual
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

# $Id$

from __future__ import division, with_statement
import itertools
import functools

import scipy
import numpy

from numina.exceptions import Error
from numina.image._combine import internal_combine as c_internal_combine

__version__ = "$Revision$"


def compressed(fun):
    '''Compressed decorator.'''
    def new(*vv, **k):
        vv = list(vv)
        vv[0] = vv[0].compressed()
        vv = tuple(vv)
        return fun(*vv, **k)
    return new

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
        '''Compute recursively the numeric intervals'''
        a = x[0]
        b = x[1]
        if b - a <= block:
            return [x]
        else:
            result = []
            d = (b - a) // 2
            temp = map(numblock, [block, block], [(a, a + d), (a + d, b)])
            for i in temp:
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
    return itertools.product(*iterables)

def py_internal_combine(method, nimages, nmasks, result, variance, numbers):
    for index, val in numpy.ndenumerate(result):
        values = [im[index] for im, mas in zip(nimages, nmasks) if not mas[index]]
        result[index], variance[index], numbers[index] = method(values)
                


def combine(method, images, masks=None, offsets=None,
            result=None, variance=None, numbers=None):
    '''Combine images using the given combine method, with masks and offsets.
    
    Inputs and masks are a list of array objects.
    
    :param method: a string with the name of the method
    :param images: a list of 2D arrays
    :param masks: a list of 2D boolean arrays, True values are masked
    :param offsets: a list of 2-tuples
    :param result: a image where the result is stored if is not None
    :param variance: a image where the variance is stored if is not None
    :param result: a image where the number is stored if is not None
    :return: a 3-tuple with the result, the variance and the number
    :raise TypeError: if method is not callable
    
    
    Example:
    
       >>> import numpy
       >>> image = numpy.array([[1.,3.],[1., -1.4]])
       >>> inputs = [image, image + 1]
       >>> combine("mean", inputs)  #doctest: +NORMALIZE_WHITESPACE
       (array([[ 1.5,  3.5],
           [ 1.5, -0.9]]), array([[ 0.25,  0.25],
           [ 0.25,  0.25]]), array([[2, 2],
           [2, 2]], dtype=int32))
       
    '''
    
    blocksize = (512, 512)
    blocksize = (2048, 2048)
    # method should be a string
    if not isinstance(method, basestring):
        raise TypeError('method is not a string')
    
    # Check inputs
    if not images:
        raise Error("len(inputs) == 0")
        
    # Offsets
    if offsets is not None:
        if len(images) != len(offsets):
            raise Error("len(inputs) != len(offsets)")
    else:
        offsets = [(0, 0)] * len(images)
    
    if masks is None:
        masks = [False] * len(images)
    else:
        if len(images) != len(masks):
            raise Error("len(inputs) != len(masks)")
        masks = map(functools.partial(numpy.asarray, dtype=numpy.bool), masks)
        for i in masks:
            t = i.dtype
            if not numpy.issubdtype(t, bool):
                raise TypeError
            if len(i.shape) != 2:
                raise TypeError
                
    aimages = map(numpy.asarray, images)
    
    for i in aimages:
        t = i.dtype
        if not (numpy.issubdtype(t, float) or numpy.issubdtype(t, int)):
            raise TypeError
        if len(i.shape) != 2:
            raise TypeError
        
    
    
    # unzip 
    # http://paddy3118.blogspot.com/2007/02/unzip-un-needed-in-python.html
    [bcor0, bcor1] = zip(*offsets)
    ucorners = [(j[0] + i.shape[0], j[1] + i.shape[1]) 
               for (i, j) in zip(aimages, offsets)]
    [ucor0, ucor1] = zip(*ucorners)
    ref = (min(bcor0), min(bcor1))
    
    finalshape = (max(ucor0) - ref[0], max(ucor1) - ref[1])
    offsetsp = [(i - ref[0], j - ref[1]) for (i, j) in offsets]
    
    if result is None:
        result = numpy.zeros(finalshape, dtype="float64")
    else:
        if result.shape != finalshape:
            raise TypeError("result has wrong shape")
    
    if variance is None:
        variance = numpy.zeros(finalshape, dtype="float64")
    else:
        if variance.shape != finalshape:
            raise TypeError("variance has wrong shape")
                
    if numbers is None:
        numbers = numpy.zeros(finalshape, dtype="int32")
    else:
        if numbers.shape != finalshape:
            raise TypeError("numbers has wrong shape")
    
    resize_images = True
    
    if resize_images:
        nimages = []
        nmasks = []
        for (i, o, m) in zip(aimages, offsetsp, masks):
            newimage1 = numpy.empty(finalshape)
            newimage2 = numpy.ones(finalshape, dtype=numpy.bool) 
            pos = (slice(o[0], o[0] + i.shape[0]), slice(o[1], o[1] + i.shape[1]))
            newimage1[pos] = i
            newimage2[pos] = m
            nimages.append(newimage1)
            nmasks.append(newimage2)
    else:
        nimages = aimages
        nmasks = masks 


    slices = [i for i in blockgen(blocksize, finalshape)]
    
    for i in slices:
        # views of the images and masks
        vnimages = [j[i] for j in nimages]
        vnmasks = [j[i] for j in nmasks]
        c_internal_combine(method, vnimages, vnmasks,
                          result[i], variance[i], numbers[i])
        
    return (result, variance, numbers)
    
if __name__ == "__main__":
    from numina.decorators import print_timing
    
    @print_timing
    def combine2(method, inputs, offsets=None):        
        return combine(method, inputs, offsets=offsets)
    
    # Inputs
    input1 = scipy.ones((2000, 2000))
    
    minputs = [input1] * 5
    minputs += [input1 * 2] * 5
    #moffsets = [(1, 1), (1, 0), (0, 0), (0, 1), (-1, -1)]
    #moffsets += [(1, 1), (1, 0), (0, 0), (0, 1), (-1, -1)]
    
    #(a,b,c) = combine2("mean", minputs, offsets=moffsets)
    (a, b, c) = combine2("mean", minputs)
    print type(a), a.dtype, a.shape, a[0, 0]
    print type(b), b.dtype, b.shape, b[0, 0]
    print type(c), c.dtype, c.shape, c[0, 0]
