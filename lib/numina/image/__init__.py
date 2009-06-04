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
from scipy import maximum, minimum

from numina.exceptions import Error

__version__ = "$Revision$"

def subarray_match(shape, ref, sshape, sref=(0, 0)):
    '''Matches two arrays, given the different shapes given by shape and shape, and a reference point ref
    It returns a tuple of slices, that can be passed to the two images as indexes
      
    Example: 
    
      >>> im = numpy.zeros((1000,1000))
      >>> sim = numpy.ones((40, 40))
      >>> i,j = subarray_match(im.shape, [20, 23], sim.shape)
      >>> im[i] = 2 * sim[j]
    
    '''
    # Reference point in im
    ref1 = numpy.array(ref, dtype='int')  
    # Ref2 is the reference point in sim  
    # The pixel [0,0] of the array by default
    ref2 = numpy.array(sref)   
    offset = ref1 - ref2    
    urc1 = minimum(offset + numpy.array(sshape) - 1, numpy.array(shape) - 1)
    blc1 = maximum(offset, 0)
    urc2 = urc1 - offset
    blc2 = blc1 - offset
    return ((slice(blc1[0], urc1[0] + 1), slice(blc1[1], urc1[1] + 1)),
            (slice(blc2[0], urc2[0] + 1), slice(blc2[1], urc2[1] + 1)))
    
    
    
def combine(method, images, masks=None, offsets=None,
            result=None, variance=None, numbers=None):
    '''Combine images using the given combine method, with masks and offsets.
    
       >>> from methods import quantileclip
       >>> import functools
       >>> method = functools.partial(quantileclip, high=2.0, low=2.5)
       >>> combine(method, [])
       
    
    '''
    # method should be a string
    
    #check images is valid
        
    if not callable(method):
        raise TypeError('method is not callable')
    
    
    
    if masks is None:
        pass

def combine2(images, masks, method="mean", args=(),
                 res=None, var=None, num=None):
    '''Combine a sequence of images using boolean masks and a combination method.
    
    Inputs and masks are a list of array objects. method can be a string or a callable object.
    args are the arguments passed to method and (res,var,num) the result
    
    '''
    
    # method should be a string
    if not isinstance(method, basestring) and not callable(method):
        raise TypeError('method is neither string nor callable')
    
    # Check inputs    
    if len(images) == 0:
        raise Error("len(inputs) == 0")
    
    if len(images) != len(masks):
        raise Error("len(inputs) != len(masks)")
    
#    def all_equal(a):
#        return all(map(lambda x: x[0] == x[1], zip(shapes,shapes[1:])))
#    # Check sizes of the images
#    shapes = [i.shape for i in images]
#    if not all_equal(shapes):
#        raise Error("shapes of inputs are different")
#
#    # Check sizes of the masks
#    shapes = [i.shape for i in masks]
#    if not all_equal(shapes):
#        raise Error("shapes of masks are different")
    
#    return test2(method, images, masks, res, var, num)

# Temporary workaround
combine1 = combine2
 



