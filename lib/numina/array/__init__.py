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

# $Id$

__version__ = "$Revision$"

import scipy as sc


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
    
      >>> im = numpy.zeros((1000, 1000))
      >>> sim = numpy.ones((40, 40))
      >>> i,j = subarray_match(im.shape, [20, 23], sim.shape)
      >>> im[i] = 2 * sim[j]
    
    '''
    # Reference point in im
    ref1 = sc.asarray(ref, dtype='int')  
    
    if sref is not None:
        ref2 = sc.asarray(sref, dtype='int')
    else:
        ref2 = sc.zeros_like(ref1) 
    
    offset = ref1 - ref2    
    urc1 = sc.minimum(offset + sc.asarray(sshape) - 1, sc.asarray(shape) - 1)
    blc1 = sc.maximum(offset, 0)
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


