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


import inspect

from .metaclass import StoreType
from .products import DataProduct, QualityAssuranceProduct

class Product(object):
    '''Product holder for RecipeResult.'''
    def __init__(self, product_type, optional=False, validate=True, 
            dest=None, *args, **kwds):

        self.validate = validate
        if inspect.isclass(product_type):
            product_type = product_type()
        self.dest = dest

        if isinstance(product_type, Optional):
            self.type = product_type.product_type
            self.optional = True
        elif isinstance(product_type, DataProduct):
            self.type = product_type
            self.optional = optional
        else:
            raise TypeError('product_type must be of class DataProduct')

    def __repr__(self):
        return 'Product(type=%r, dest=%r)' % (self.type, self.dest)


class Optional(object):
    def __init__(self, product_type):

        if inspect.isclass(product_type):
            product_type = product_type()

        if isinstance(product_type, DataProduct):
            self.type = product_type
        else:
            raise TypeError('product_type must be of class DataProduct')

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
        return "%s(errortype=%r, message='%s')" % (sclass, 
            self.errortype, self.message)

class RecipeResultType(StoreType):
    '''Metaclass for RecipeResult.'''
    @classmethod
    def exclude(cls, value):
        return isinstance(value, Product)

class RecipeResultAutoQAType(RecipeResultType):
    '''Metaclass for RecipeResult with added QA'''
    def __new__(cls, classname, parents, attributes):
        if 'qa' not in attributes:
            attributes['qa'] = Product(QualityAssuranceProduct)
        return super(RecipeResultAutoQAType, cls).__new__(cls, classname, parents, attributes)

class RecipeResult(BaseRecipeResult):
    __metaclass__ = RecipeResultType

    def __new__(cls, *args, **kwds):
        self = super(RecipeRecipeResult, cls).__new__(cls)
        for key, prod in self.__stored__.iteritems():
            if key in kwds:
                # validate
                val = kwds[key]
                if prod.validate:
                    prod.type.validate(val)
                val = prod.type.store(val)
                setattr(self, key, val)
            elif prod.type.default:
                val = prod.type.default
                if prod.validate:
                    prod.type.validate(val)
                val = prod.type.store(val)
                setattr(self, key, val)
            elif not prod.optional:
                raise ValueError('required DataProduct %r not defined' % prod.type.__class__.__name__)
            else:
                # optional product, skip
                setattr(self, key, None)
        return self

    def __init__(self, *args, **kwds):
        super(RecipeResult, self).__init__(self, *args, **kwds)

    def __repr__(self):
        sclass = type(self).__name__
        full = []
        for key, val in self.__stored__.iteritems():
            full.append('%s=%r' % (key, val))
        return '%s(%s)' % (sclass, ', '.join(full))

    def suggest_store(self, **kwds):
        for k in kwds:
            mm = getattr(self, k)
            self.__stored__[k].type.suggest(mm, kwds[k])

class RecipeResultAutoQA(RecipeResult):
    '''RecipeResult with an automatic QA member.'''
    __metaclass__ = RecipeResultAutoQAType

def transmit(result):
    if not isinstance(result, BaseRecipeResult):
        raise TypeError('result must be a RecipeResult')
    if isinstance(result, RecipeResult):
        pass # transmit as valid'
    elif isinstance(result, ErrorRecipeResult):
        res = {'error': {'type': result.errortype,
                         'message': result.message,
                         'traceback': result.traceback}
                         }
        return res
    else:
        raise TypeError('Unknown subclass of RecipeResult')

class define_result(object):
    def __init__(self, resultClass):
        if not issubclass(resultClass, BaseRecipeResult):
            raise TypeError('%r does not derive from BaseRecipeResult' % resultClass)

        self.provides = []
        self.klass = resultClass

        for key, val in resultClass.__stored__.iteritems():
            val.dest = key
            if isinstance(val, Product):
                self.provides.append(val)

    def __call__(self, klass):
        klass.__provides__ = self.provides

        klass.RecipeResult = self.klass
        return klass

