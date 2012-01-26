#
# Copyright 2010-2012 Universidad Complutense de Madrid
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

'''Iterate over a Queue.'''

import Queue

def iterqueue(qu):
    '''Iterate over a Queue.
    
    The iteration doesn't block. It calls Queue.get_nowait method
    
    Example usage:
    
    >>> qu = Queue.Queue()
    >>> qu.put(1)
    >>> qu.put(2)
    >>> qu.put(3)    
    >>> for i in iterqueue(qu):
    ...     print i
    ... 
    1
    2
    3
    '''
    try:
        while True:
            yield qu.get_nowait()
    except Queue.Empty:
        raise StopIteration
