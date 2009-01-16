#
# Copyright 2008 Sergio Pascual
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

def new_combine(inputs, masks, method, args, res=None, var=None, num=None):
    '''Inputs and masks are a list of array objects. method can be a string or a callable object
    args are the arguments passed to method and (res,var,num) the result'''
    
    # Check inputs
    
    if length(inputs) == 0:
      raise Error
    
    if length(inputs) != length(masks):
      raise Error
    