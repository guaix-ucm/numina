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

'''
Recipe requirements
'''

from .requirements import Requirement

class RecipeRequirementsType(type):
    '''Metaclass for RecipeRequirements.'''
    def __new__(cls, classname, parents, attributes):
        filter_out = {}
        filter_in = {}
        filter_in['__stored__'] = filter_out
        for name, val in attributes.items():
            if isinstance(val, Requirement):
                filter_out[name] = val
            else:
                filter_in[name] = val
        return super(RecipeRequirementsType, cls).__new__(cls, classname, parents, filter_in)

    def __setattr__(cls, key, value):
        cls._add_attr(key, value)

    def _add_attr(cls, key, val):
        if isinstance(val, Requirement):
            cls.__stored__[key] = val
        else:
            super(RecipeRequirementsType, cls).__setattr__(key, value)

class RecipeRequirements(object):
    '''RecipeRequirements base class'''
    __metaclass__ = RecipeRequirementsType
    def __new__(cls, *args, **kwds):
        self = super(RecipeRequirements, cls).__new__(cls)
        for key, req in cls.__stored__.items():
            if key in kwds:
                # validate
                val = kwds[key]
                #if req.validate:
                #    req.type.validate(val)
                val = req.type.store(val)
                setattr(self, key, val)
            elif not req.optional:
                raise ValueError(' %r of type %r not defined' % (key, req.type.__class__.__name__))
            else:
                # optional product, skip
                setattr(self, key, None)
        return self

    def __init__(self, *args, **kwds):
        super(RecipeRequirements, self).__init__()

class define_requirements(object):
    def __init__(self, requirementClass):
        if not issubclass(requirementClass, RecipeRequirements):
            raise TypeError('%r does not derive from RecipeRequirements' % requirementClass)

        self.klass = requirementClass
        self.requires = []

        for key, val in requirementClass.__stored__.items():
            if isinstance(val, Requirement):
                if val.dest is None:
                    val.dest = key
            self.requires.append(val)

    def __call__(self, klass):
        klass.__requires__ = self.requires
        klass.RecipeRequirements = self.klass
        return klass

