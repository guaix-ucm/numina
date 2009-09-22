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

import numpy
import numpy.ma as ma

from numina.exceptions import Error

__version__ = "$Revision$"


def compressed(fun):
    '''Compressed decorator.'''
    def new(*vv, **k):
        vv = list(vv)
        vv[0] = vv[0].compressed()
        vv = tuple(vv)
        return fun(*vv, **k)
    return new

def blockgen(shape, block):
    '''Generate 1d block intervals to be used by combine.'''
    low = 0
    high = block
    while(low < shape):
        yield (low, max(high, shape))
        low , high = high, high + block


def combine(method, images, masks=None, offsets=None,
            result=None, variance=None, numbers=None):
    '''Combine images using the given combine method, with masks and offsets.
    
    Inputs and masks are a list of array objects.
    
    :param method: a callable object that accepts a sequence.
    :param images: a list of 2D arrays
    :param masks: a list of 2D boolean arrays, True values are masked
    :param offsets: a list of 2-tuples
    :param result: a image where the result is stored if is not None
    :param variance: a image where the variance is stored if is not None
    :param result: a image where the number is stored if is not None
    :return: a 3-tuple with the result, the variance and the number
    :raise TypeError: if method is not callable
    
    
    Example:
    
       >>> from methods import quantileclip
       >>> import functools
       >>> method = functools.partial(quantileclip, high=2.0, low=2.5)
       >>> import numpy
       >>> image = numpy.array([[1.,3.],[1., -1.4]])
       >>> inputs = [image, image + 1]
       >>> combine(method, inputs)  #doctest: +NORMALIZE_WHITESPACE
       (array([[ 0.,  0.], [ 0.,  0.]]), 
        array([[ 0.,  0.], [ 0.,  0.]]), 
        array([[ 0.,  0.], [ 0.,  0.]]))
       
    
    '''
    # method should be callable
    # if not isinstance(method, basestring)
    if not callable(method):
        raise TypeError('method is not callable')
    
    # Check inputs
    if len(images) == 0:
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
        
        
    # unzip 
    # http://paddy3118.blogspot.com/2007/02/unzip-un-needed-in-python.html
    [bcor0, bcor1] = zip(*offsets)
    ucorners = [(j[0] + i.shape[0], j[1] + i.shape[1]) 
               for (i,j) in zip(images, offsets)]
    [ucor0, ucor1] = zip(*ucorners)
    ref = (min(bcor0), min(bcor1))
    
    finalshape = (max(ucor0) - ref[0], max(ucor1) - ref[1])
    offsetsp = [(i - ref[0], j - ref[1]) for (i, j) in offsets]
    
    if result is None:
        result = numpy.zeros(finalshape)
    else:
        if result.shape != finalshape:
            raise TypeError("result has wrong shape")
    
    if variance is None:
        variance = numpy.zeros(finalshape)
    else:
        if variance.shape != finalshape:
            raise TypeError("variance has wrong shape")
                
    if numbers is None:
        numbers = numpy.zeros(finalshape)
    else:
        if numbers.shape != finalshape:
            raise TypeError("numbers has wrong shape")
    
    r = []
    for (i, o, m) in zip(images, offsetsp, masks):
        newimage = ma.array(numpy.zeros(finalshape), mask=True) 
        pos = (slice(o[0], o[0] + i.shape[0]), slice(o[1], o[1] + i.shape[1]))
        newimage[pos] = ma.array(i, mask=m)
        r.append(newimage)

    cube = ma.array(r)
#    print "cube shape", cube.shape
    method = compressed(method)
    val = ma.apply_along_axis(method, 0, cube)
    
    
    result = val.data[0]
    variance = val.data[1]
    numbers = val.data[2]
    
    return (val.data[0], val.data[1], val.data[2])
    