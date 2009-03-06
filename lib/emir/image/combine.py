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

from emir.exceptions import Error
from _combine import test1
from _combine import test2
from _combine import method_mean

__version__ = "$Id$"

def new_combine2(images, masks, method="mean", args=(), res=None, var=None, num=None):
    '''Inputs and masks are a list of array objects. method can be a string or a callable object
    args are the arguments passed to method and (res,var,num) the result'''
    
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
    
    return test2(method, images, masks, res, var, num)

# Temporary workaround
new_combine1 = new_combine2
    