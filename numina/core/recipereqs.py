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
        requirements = {}
        filter_in = {}
        filter_in['_requirements'] = requirements
        for name,val in attributes.items():
            if isinstance(val, Requirement):
                requirements[name] = val
            else:
                filter_in[name] = val
        return super(MyType, cls).__new__(cls, classname, parents, filter_in)

    def __setattr__(cls, key, value):
        cls._add_attr(key, value)

    def _add_attr(cls, key, val):
        if isinstance(val, Requirement):
            cls._requirements[key] = val
        else:
            super().__setattr__(key, value)

class RecipeRequirementsBase(object):
    '''RecipeRequirements base class'''
    __metaclass__ = MyType
    def __new__(cls, *args, **kwds):
        self = super(Base, cls).__new__(cls)
        for key, req in cls._requirements.items():
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
        super().__init__()

class RecipeRequirements(object):
    def __new__(cls, *args, **kwds):
        cls._requirements = {}
        for name in dir(cls):
            if not name.startswith('_'):
                val = getattr(cls, name)
                if isinstance(val, Requirement):
                    cls._requirements[name] = val

        return super(RecipeRequirements, cls).__new__(cls)

    def __init__(self, *args, **kwds):
        for key, req in self._requirements.iteritems():
            if key in kwds:
                # validate
                val = kwds[key]
                #if req.validate:
                #    req.type.validate(val)
                val = req.type.store(val)
                setattr(self, key, val)
            elif not req.optional:
                raise ValueError(' %r not defined' % req.type.__class__.__name__)
            else:
                # optional product, skip
                setattr(self, key, None)

        super(RecipeRequirements, self).__init__(self, *args, **kwds)

class requires(object):
    '''Decorator to add the list of required parameters to recipe'''
    def __init__(self, *requirements):
        self.requirements = requirements

    def __call__(self, klass):
        if hasattr(klass, '__requires__'):
            klass.__requires__.extend(self.requirements)
        else:
            klass.__requires__ = list(self.requirements)
        return klass

class define_requirements(object):
    def __init__(self, requirementClass):
        if not issubclass(requirementClass, RecipeRequirements):
            raise TypeError

        self.klass = requirementClass
        self.requires = []

        for i in dir(requirementClass):
            if not i.startswith('_'):
                val = getattr(requirementClass, i)
                if isinstance(val, Requirement):
                    if val.dest is None:
                        val.dest = i
                    self.requires.append(val)

    def __call__(self, klass):
        klass.__requires__ = self.requires
        klass.RecipeRequirements = self.klass
        return klass
