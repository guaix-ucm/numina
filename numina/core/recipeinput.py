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
Recipe inputs
'''

from .requirements import Requirement

class RecipeInput(object):
    def __new__(cls, *args, **kwds):
        cls._requirements = {}
        for name in dir(cls):
            if not name.startswith('_'):
                val = getattr(cls, name)
                if isinstance(val, Requirement):
                    cls._products[name] = val

        return super(RecipeInput, cls).__new__(cls, *args, **kwds)

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

class define_input(object):
    def __init__(self, input_):
        if not issubclass(input_, RecipeInput):
            raise TypeError

        self.klass = input_
        self.requires = []

        for i in dir(input_):
            if not i.startswith('_'):
                val = getattr(input_, i)
                if isinstance(val, Requirement):
                    if val.dest is None:
                        val.dest = i
                    self.requires.append(val)

    def __call__(self, klass):
        klass.__requires__ = self.requires
        klass.RecipeInput = self.klass
        return klass
