#
# Copyright 2008-2011 Sergio Pascual
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

'''Numina data processing system.'''

import logging

# pylint: disable-msg=E0611
try:
    from logging import NullHandler
except ImportError:
    from logger import NullHandler

__version__ = '0.3.0'

# Top level NullHandler
logging.getLogger("numina").addHandler(NullHandler())

def braid(*iterables):
    '''Return the elements of each iterator in turn until some is exhausted.
    
    This function is similar to the roundrobin example 
    in itertools documentation.
    
    >>> a = iter([1,2,3,4])
    >>> b = iter(['a', 'b'])
    >>> c = iter([1,1,1,1,'a', 'c'])
    >>> d = iter([1,1,1,1,1,1])
    >>> list(braid(a, b, c, d))
    [1, 'a', 1, 1, 2, 'b', 1, 1]
    '''
    
    from itertools import izip
    
    for itbl in izip(*iterables):
        for it in itbl:
            yield it
