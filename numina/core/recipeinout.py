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
Recipe Input
"""

from six import with_metaclass

from .metaclass import RecipeInputType, RecipeResultType


class RecipeInOut(object):
    def __new__(cls, *args, **kwds):
        self = super(RecipeInOut, cls).__new__(cls)
        for key, prod in cls.stored().items():
            if key in kwds:
                val = prod.convert(kwds[key])
            else:
                # Value not defined
                val = prod.default_value()

            if prod.choices and (val not in prod.choices):
                raise ValueError('%s not in %s' % (val, prod.choices))

            setattr(self, key, val)
        return self

    def __init__(self, *args, **kwds):
        super(RecipeInOut, self).__init__()

    def validate(self):
        '''Validate myself.'''

        # By default, validate each value
        for key, req in self.__class__.items():
            val = getattr(self, key)
            req.validate(val)

        # Run checks defined in __checkers__
        self._run_checks()

    def _run_checks(self):
        checkers = getattr(self, '__checkers__', [])

        for check in checkers:
            check.check(self)


class RecipeInput(with_metaclass(RecipeInputType, RecipeInOut)):
    """RecipeInput base class"""
    pass


class BaseRecipeResult(object):
    def __new__(cls, *args, **kwds):
        return super(BaseRecipeResult, cls).__new__(cls)

    def __init__(self, *args, **kwds):
        super(BaseRecipeResult, self).__init__()


class ErrorRecipeResult(BaseRecipeResult):
    def __init__(self, errortype, message, traceback):
        self.errortype = errortype
        self.message = message
        self.traceback = traceback

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(errortype=%r, message='%s')"
        return fmt % (sclass, self.errortype, self.message)


class RecipeResult(with_metaclass(RecipeResultType, RecipeInOut, BaseRecipeResult)):

    def __repr__(self):
        sclass = type(self).__name__
        full = []
        for key, val in self.__class__.items():
            full.append('%s=%r' % (key, val))
        return '%s(%s)' % (sclass, ', '.join(full))


class define_result(object):
    '''Recipe decorator.'''
    def __init__(self, resultClass):
        if not issubclass(resultClass, RecipeResult):
            msg = '%r does not derive from RecipeResult' % resultClass
            raise TypeError(msg)
        self.klass = resultClass

    def __call__(self, klass):
        klass.Result = self.klass
        # TODO: remove this name in the future
        klass.RecipeResult = self.klass
        return klass


class define_input(object):
    '''Recipe decorator.'''
    def __init__(self, inputClass):
        if not issubclass(inputClass, RecipeInput):
            fmt = '%r does not derive from RecipeInput'
            msg = fmt % inputClass
            raise TypeError(msg)
        self.klass = inputClass

    def __call__(self, klass):
        klass.Input = self.klass
        # TODO: remove this name in the future
        klass.RecipeInput = self.klass
        return klass


define_requirements = define_input


class add_requirement(object):
    def __init__(self, **kwds):

        self.ext = {}
        for key, val in kwds.items():
            # FIXME validate these inputs
            self.ext[key] = val

    def __call__(self, klass):
        Class = klass.RecipeInput
        Class.__stored__.update(self.ext)
        return klass


class add_product(object):
    def __init__(self, **kwds):
        # FIXME validate these inputs
        self.ext = {}
        for key, val in kwds.items():
            # FIXME validate these inputs
            self.ext[key] = val

    def __call__(self, klass):
        Class = klass.RecipeResult
        Class.__stored__.update(self.ext)
        return klass