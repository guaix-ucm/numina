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

from .metaclass import RecipeRequirementsType, RecipeResultType
from .metaclass import RecipeResultAutoQCType


class RecipeInOut(object):
    def __new__(cls, *args, **kwds):
        self = super(RecipeInOut, cls).__new__(cls)
        for key, prod in cls.iteritems():
            store_val = True
            if key in kwds:
                val = kwds[key]
            else:
                # Value not defined
                if prod.default is not None:
                    val = prod.default
                elif prod.type.default is not None:
                    val = prod.type.default
                elif prod.optional:
                    val = None
                    store_val = False
                else:
                    fmt = 'Required %r of type %r not defined'
                    msg = fmt % (key, prod.type)
                    raise ValueError(msg)

            if store_val:
                nval = prod.type.store(val)
            else:
                nval = val

            if prod.choices and (nval not in prod.choices):
                raise ValueError('%s not in %s' % (nval, prod.choices))

            setattr(self, key, nval)
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


class RecipeRequirements(RecipeInOut):
    '''RecipeRequirements base class'''
    __metaclass__ = RecipeRequirementsType


class BaseRecipeResult(object):
    def __new__(cls, *args, **kwds):
        return super(BaseRecipeResult, cls).__new__(cls)

    def __init__(self, *args, **kwds):
        super(BaseRecipeResult, self).__init__()

    def suggest_store(self, *args, **kwds):
        pass


class ErrorRecipeResult(BaseRecipeResult):
    def __init__(self, errortype, message, traceback):
        self.errortype = errortype
        self.message = message
        self.traceback = traceback

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(errortype=%r, message='%s')"
        return fmt % (sclass, self.errortype, self.message)


class RecipeResult(RecipeInOut, BaseRecipeResult):
    __metaclass__ = RecipeResultType

    def __repr__(self):
        sclass = type(self).__name__
        full = []
        for key, val in self.__class__.iteritems():
            full.append('%s=%r' % (key, val))
        return '%s(%s)' % (sclass, ', '.join(full))

    def suggest_store(self, **kwds):
        for k in kwds:
            mm = getattr(self, k)
            self.__class__[k].type.suggest(mm, kwds[k])


class RecipeResultAutoQC(RecipeResult):
    '''RecipeResult with an automatic QC member.'''
    __metaclass__ = RecipeResultAutoQCType


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


class define_requirements(object):
    '''Recipe decorator.'''
    def __init__(self, requirementClass):
        if not issubclass(requirementClass, RecipeRequirements):
            fmt = '%r does not derive from RecipeRequirements'
            msg = fmt % requirementClass
            raise TypeError(msg)
        self.klass = requirementClass

    def __call__(self, klass):
        klass.Requirements = self.klass
        # TODO: remove this name in the future
        klass.RecipeRequirements = self.klass
        return klass
