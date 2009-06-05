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

import numpy.ma as ma

from numina.exceptions import Error

__version__ = "$Revision$"


def compressed(fun):
    '''compressed decorator'''
    def new(*a, **k):
        a[0] = a[0].compressed()
        return fun(*a, **k)
    return new

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
       >>> combine(method, [])
       
    
    '''
    # method should be callable
    # if not isinstance(method, basestring)
    if not callable(method):
        raise TypeError('method is not callable')
    
    # Check inputs
    if len(images) == 0:
        raise Error("len(inputs) == 0")
    
    if len(images) != len(masks):
        raise Error("len(inputs) != len(masks)") 
    
    if masks is None:
        return ()
    else:
        method = compressed(method)
        r = [ma.array(i, mask=j) for (i,j) in zip(images, masks)]
        r = ma.dstack(r)
        val = ma.apply_along_axis(method, 0, r)
        data = val.data
        return (data[0], data[1], data[2])
        
    
    
    
    
    
        