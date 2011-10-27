#
# Copyright 2008-2011 Sergio Pascual
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

'''Very simple generic function implementation.'''

class GenericFunction(object):
    def __init__(self, f):
        self._internal_map = {}
        self._default_impl = f

    def register(self, cls):
        '''Register a new type class with a generic function.
        
        Classes managed by generic must be new type
        >>> @generic
        ... def store(obj, where):
        ...     print 'object not storable'
        ... 
        >>> class B(object): 
        ...     pass
        ...
        >>> @store.register(B)
        ... def store_b(obj, where):
        ...     print 'Storing a B object in', where
        ...
        >>> b = B()
        >>> store(b, 'somewhere')
        Storing a B object in somewhere
        
        '''

        def decorator(f):
            self._internal_map[cls] = f
            return f

        return decorator

    def unregister(self, cls):
        '''Remove a registered class within a generic function.
        
        >>> @generic
        ... def store(obj, where):
        ...     print 'object not storable'
        ... 
        >>> class B(object): 
        ...     pass
        ...
        >>> @store.register(B)
        ... def store_b(obj, where):
        ...     print 'Storing a B object in', where
        ...
        >>> store.unregister(B)
        >>> b = B()
        >>> store(b, 'somewhere')
        object not storable
        
        '''

        if cls in self._internal_map:
            del self._internal_map[cls]


    def __call__(self, *args):
        obj = args[0]
        candidates = []
        for cls in self._internal_map:
            if issubclass(obj.__class__, cls):
                candidates.append(cls)
        for base in obj.__class__.__mro__:
            if base in candidates:
                func = self._internal_map[base]
                return func(*args)
        else:
            self._default_impl(*args)
            
    def is_registered(self, cls):
        if cls in self._internal_map:
            return True
        return False

def generic(function):
    '''Decorate a function, making it the default implementation of a generic function.
    
    >>> @generic
    ... def store(obj, where):
    ...     print 'object not storable'
    ... 
    >>> store('something', 'somewhere')
    object not storable
    
    Classes managed by generic must be new type
    >>> class B(object): 
    ...     pass
    ...
    >>> @store.register(B)
    ... def store_b(obj, where):
    ...     print 'Storing a B object in', where
    ...
    >>> b = B()
    >>> store(b, 'somewhere')
    Storing a B object in somewhere
    
    generic follows inheritance diagram
    >>> class C(B):
    ...    pass
    ...
    >>> c = C()
    >>> store(c, 'somewhere')
    Storing a B object in somewhere
    '''
    return GenericFunction(function)


