#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

'''
Base metaclasses
'''

import collections

class StoreType(type):
    '''Metaclass for storing members.'''
    def __new__(cls, classname, parents, attributes):
        filter_out = {}
        filter_in = {}
        filter_in['__stored__'] = filter_out
        # Handle stored values from parents
        for p in parents:
            stored = getattr(p, '__stored__', None)
            if stored:
                filter_in['__stored__'].update(stored)

        for name, val in attributes.items():
            if cls.exclude(val):
                filter_out[name] = cls.store(name, val)
            else:
                filter_in[name] = val
        return super(StoreType, cls).__new__(cls, classname, parents, filter_in)

    def __setattr__(self, key, value):
        self._add_attr(key, value)

    def _add_attr(self, key, value):
        if self.exclude(value):
            self.__stored__[key] = value
        else:
            super(StoreType, cls).__setattr__(key, value)

    @classmethod
    def exclude(cls, value):
        return False

    @classmethod
    def store(cls, name, value):
        return value


class MapStoreType(StoreType, collections.Mapping):
    '''Metaclass for storing members with map interface.'''
    def __new__(cls, classname, parents, attributes):
        return super(MapStoreType, cls).__new__(cls, classname, parents, attributes)
    
    def __getitem__(self, name):
        return self.__stored__[name]

    def __iter__(self):
        return iter(self.__stored__)

    def __len__(self):
        return len(self.__stored__)

