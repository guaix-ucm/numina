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
from .products import Product
from .requirements import Requirement
from .datadescriptors import QualityControlProduct

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
            if cls.exclude(name, val):
                nname, nval = cls.store(name, val)
                filter_out[nname] = nval
            else:
                filter_in[name] = val
        return super(StoreType, cls).__new__(cls, classname, parents, filter_in)

    def __setattr__(self, key, value):
        self._add_attr(key, value)

    def _add_attr(self, key, value):
        if self.exclude(key, value):
            self.__stored__[key] = value
        else:
            super(StoreType, cls).__setattr__(key, value)

    @classmethod
    def exclude(cls, name, value):
        return False

    @classmethod
    def store(cls, name, value):
        return name, value

# FIXME: this does not work due to this
# http://comments.gmane.org/gmane.comp.python.devel/142467
#class MapStoreType(StoreType, collections.Mapping)
# Manual impl instead 

class MapStoreType(StoreType):
    '''Metaclass for storing members with map interface.'''
    def __new__(cls, classname, parents, attributes):
        return super(MapStoreType, cls).__new__(cls, classname, parents, attributes)
    
    def __getitem__(self, name):
        return self.__stored__[name]

    def __iter__(self):
        return iter(self.__stored__)

    def __len__(self):
        return len(self.__stored__)

    # These methods are implemented from the previous
    # Copied over from collections
    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def iterkeys(self):
        'D.iterkeys() -> an iterator over the keys of D'
        return iter(self)

    def itervalues(self):
        'D.itervalues() -> an iterator over the values of D'
        for key in self:
            yield self[key]

    def iteritems(self):
        'D.iteritems() -> an iterator over the (key, value) items of D'
        for key in self:
            yield (key, self[key])

    def keys(self):
        "D.keys() -> list of D's keys"
        return list(self)

    def items(self):
        "D.items() -> list of D's (key, value) pairs, as 2-tuples"
        return [(key, self[key]) for key in self]

    def values(self):
        "D.values() -> list of D's values"
        return [self[key] for key in self]

    # Mappings are not hashable by default, but subclasses can change this
    __hash__ = None

    def __eq__(self, other):
        if not isinstance(other, collections.Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __ne__(self, other):
        return not (self == other)

class RecipeInOuttType(MapStoreType):    
    @classmethod
    def store(cls, name, value):
        if value.dest is None:
            value.dest = name
        nname = value.dest
        return nname, value

class RecipeRequirementsType(RecipeInOuttType):
    '''Metaclass for RecipeRequirements.'''

    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Requirement)
    
class RecipeResultType(RecipeInOuttType):
    '''Metaclass for RecipeResult.'''
    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Product)

class RecipeResultAutoQCType(RecipeResultType):
    '''Metaclass for RecipeResult with added QC'''
    def __new__(cls, classname, parents, attributes):
        if 'qc' not in attributes:
            attributes['qc'] = Product(QualityControlProduct)
        return super(RecipeResultAutoQCType, cls).__new__(cls, classname, parents, attributes)


