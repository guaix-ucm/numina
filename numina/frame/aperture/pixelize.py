#
# Copyright 2008-2013 Universidad Complutense de Madrid
# 
# This file is part of Numina
# 
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
# 

import math

def frac(x):
    '''Return the fractional part of a real number'''
    return x - math.trunc(x)

def pixelize(x1, x2):
    '''Generate tuples of values inside pixels
    
    >>> list(pixelize(1, 0))
    []
    >>> list(pixelize(1.2, 1.2))
    []
    >>> list(pixelize(1.0, 3.0))
    [(1.0, 2.0), (2.0, 3.0)]
    >>> list(pixelize(1.1, 3.0))
    [(1.1, 2.0), (2.0, 3.0)]
    >>> list(pixelize(1.0, 3.2))
    [(1.0, 2.0), (2.0, 3.0), (3.0, 3.2)]
    >>> list(pixelize(1.1, 3.2))
    [(1.1, 2.0), (2.0, 3.0), (3.0, 3.2)]
    '''
        
    if x2 <= x1:
        return

    l = x1
    u = math.floor(x1) + 1.0

    while u < x2:
        yield l, u
        l, u = u, u + 1.0

    yield l, x2

def rpixelize(x1, x2):
    '''Generate tuples of values inside pixels in reverse order
    
    >>> list(rpixelize(1, 0))
    []
    >>> list(rpixelize(1.2, 1.2))
    []
    >>> list(rpixelize(1.0, 3.0))
    [(2.0, 3.0), (1.0, 2.0)]
    >>> list(rpixelize(1.1, 3.0))
    [(2.0, 3.0), (1.1, 2.0)]
    >>> list(rpixelize(1.0, 3.2))
    [(3.0, 3.2), (2.0, 3.0), (1.0, 2.0)]
    >>> list(rpixelize(1.1, 3.2))
    [(3.0, 3.2), (2.0, 3.0), (1.1, 2.0)]
    '''
    if x2 <= x1:
        return
    
    u = x2
    l = math.ceil(x2) - 1.0

    while l > x1:
        yield l, u
        l, u = l - 1.0, l

    yield x1, u

