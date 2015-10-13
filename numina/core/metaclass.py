#
# Copyright 2008-2015 Universidad Complutense de Madrid
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

"""
Base metaclasses
"""

from .dataholders import Product
from .requirements import Requirement
import weakref


class StoreType(type):
    """Metaclass for storing members."""
    def __new__(cls, classname, parents, attributes):

        n_stored = weakref.WeakValueDictionary()
        for p in parents:
            stored = getattr(p, '__numina_stored__', None)
            if stored:
                n_stored.update(stored)

        for name, val in attributes.items():
            if cls.exclude(name, val):
                nname, nval = cls.transform(name, val)
                n_stored[nname] = nval

        attributes['__numina_stored__'] = n_stored

        return super(StoreType, cls).__new__(cls, classname, parents, attributes)

    def __setattr__(self, key, value):
        """Define __setattr__ in 'classes' created with this metaclass."""
        self._add_attr(key, value)

    def _add_attr(self, key, value):
        if self.exclude(key, value):
            nkey, nvalue = self.transform(key, value)
            self.__numina_stored__[nkey] = nvalue

        super(StoreType, self).__setattr__(key, value)

    @classmethod
    def exclude(cls, name, value):
        return False

    @classmethod
    def transform(cls, name, value):
        return name, value


class RecipeInOutType(StoreType):
    def __new__(cls, classname, parents, attributes):
        # Handle checkers defined in base class
        checkers = attributes.get('__checkers__', [])
        for p in parents:
            c = getattr(p, '__checkers__', [])
            checkers.extend(c)
        attributes['__checkers__'] = checkers
        obj = super(RecipeInOutType, cls).__new__(
            cls, classname, parents, attributes)
        return obj

    @classmethod
    def transform(cls, name, value):
        if value.dest is None:
            value.dest = name
        nname = value.dest
        return nname, value


class RecipeInputType(RecipeInOutType):
    """Metaclass for RecipeInput."""
    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Requirement)


class RecipeResultType(RecipeInOutType):
    """Metaclass for RecipeResult."""
    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Product)
