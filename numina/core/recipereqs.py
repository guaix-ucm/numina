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
Recipe requirements
'''

from .metaclass import MapStoreType
from .requirements import Requirement

class RecipeRequirementsType(MapStoreType):
    '''Metaclass for RecipeRequirements.'''

    @classmethod
    def exclude(cls, name, value):
        return isinstance(value, Requirement)

    @classmethod
    def store(cls, name, value):
        if value.dest is None:
            value.dest = name
        return value

class RecipeRequirements(object):
    '''RecipeRequirements base class'''
    __metaclass__ = RecipeRequirementsType
    def __new__(cls, *args, **kwds):
        self = super(RecipeRequirements, cls).__new__(cls)
        for key, req in cls.iteritems():
            if key in kwds:
                val = kwds[key]
            else:
                # Value not defined...
                if req.default is not None:
                    val = req.default
                elif req.type.default is not None:
                    val = req.type.default
                elif req.optional:
                    val = None
                else:
                    raise ValueError(' %r of type %r not defined' % (key, req.type))
                
            val = req.type.store(val)
            
            if req.choices and (val not in req.choiches):
                raise ValueError('%s not in %s' % (val, req.choices))

            if req.validate:
                valid = req.type.validate(val)
                if not valid:
                    raise ValueError(' %r of type %r not valid' % (val, req.type.python_type))
            setattr(self, key, val)
            
        return self

    def __init__(self, *args, **kwds):
        super(RecipeRequirements, self).__init__()

class define_requirements(object):
    def __init__(self, requirementClass):
        if not issubclass(requirementClass, RecipeRequirements):
            raise TypeError('%r does not derive from RecipeRequirements' % requirementClass)

        self.klass = requirementClass
        self.requires = []

        for key, val in requirementClass.iteritems():
            self.requires.append(val)

    def __call__(self, klass):
        klass.__requires__ = self.requires
        klass.RecipeRequirements = self.klass
        return klass

