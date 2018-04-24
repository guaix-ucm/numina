#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Base metaclasses
"""

from .dataholders import Result
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

        new_attributes = {}
        for name, val in attributes.items():
            if cls.exclude(name, val):
                new_name, new_val = cls.transform(name, val)
                n_stored[new_name] = new_val
                new_attributes[new_name] = new_val
            else:
                new_attributes[name] = val

        new_attributes['__numina_stored__'] = n_stored

        return super(StoreType, cls).__new__(cls, classname, parents, new_attributes)

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
        if value.dest is None:
            value.dest = name
        nname = value.dest
        return nname, value


class RecipeInputType(StoreType):
    """Metaclass for RecipeInput."""
    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Requirement)


class RecipeResultType(StoreType):
    """Metaclass for RecipeResult."""
    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Result)
