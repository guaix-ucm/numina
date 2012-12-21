#
# Copyright 2008-2012 Universidad Complutense de Madrid
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
                # FIXME: better create a DefaultType
                # that leaves value unchanged
                if req.type is not None:
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
