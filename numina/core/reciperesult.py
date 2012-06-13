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


from .product import Product
import traceback

class RecipeResult(object):
    def __news__(cls):
        return super(RecipeResult, cls).__new__(cls)

class ErrorRecipeResult(RecipeResult):
    def __init__(self, errortype, message, traceback):
        self.errortype = errortype
        self.message = message
        self.traceback = traceback

    def __repr__(self):
        sclass = type(self).__name__
        return "%s(errortype=%r, message='%s')" % (sclass, 
            self.errortype, self.message)

class ValidRecipeResult(RecipeResult):

    def __new__(cls, *args, **kwds):

        cls._products = {}
        for key, val in cls.__dict__.iteritems():
            if isinstance(val, Product):
                cls._products[key] = val

        return super(ValidRecipeResult, cls).__new__(cls, *args, **kwds)

    def __init__(self, *args, **kwds):
        for key, prod in self._products.iteritems():
            if key in kwds:
                # validate
                val = kwds[key]
                prod.product_type.validate(val)
                val = prod.product_type.store(val)
                setattr(self, key, val)
            elif not prod.optional:
                raise ValueError('required DataProduct %r not defined' % prod.product_type.__class__.__name__)
            else:
                # optional product, skip
                setattr(self, key, None)

        super(ValidRecipeResult, self).__init__(self, *args, **kwds)

    def __repr__(self):
        sclass = type(self).__name__
        full0 = '%s(' % sclass
        full = []
        for key in self._products:
            val = getattr(self, key)
            full.append('%s=%r' % (key, val))
        return '%s(%s)' % (sclass, ', '.join(full))

def transmit(result):
    if not isinstance(result, RecipeResult):
        raise TypeError('result must be a RecipeResult')
    if isinstance(result, ValidRecipeResult):
        print 'transmit as valid'
    elif isinstance(result, ErrorRecipeResult):
        res = {'error': {'type': result.errortype,
                         'message': result.message,
                         'traceback': result.traceback}
                         }
        return res
    else:
        raise TypeError('Unknown subclass of RecipeResult')

class define_result(object):
    def __init__(self, result):
        if not issubclass(result, ValidRecipeResult):
            raise TypeError

        self.provides = {}
        self.klass = result

        for i in dir(result):
            if not i.startswith('_'):
                val = getattr(result, i)
                if isinstance(val, Product):
                    # provides should be a dictionary...
                    #print i, val.product_type, val.optional
                    self.provides[i] = val.product_type.__class__

    def __call__(self, klass):
        if not hasattr(klass, '__provides__'):
            klass.__provides__ = self.provides

        klass.RecipeResult = self.klass
        return klass

class provides(object):
    '''Decorator to add the list of provided products to recipe'''
    def __init__(self, *products, **kwds):
        prods = kwds
        basename = 'product%d'
        for i, prod in enumerate(products):
            prods[basename % i] = prod
        self.products = {k: Product(v) for k, v in prods.iteritems()}

    def __call__(self, klass):
        if hasattr(klass, '__provides__'):
            klass.__provides__.update(self.products)
        else:
            klass.__provides__ = self.products

        a = type('%s_RecipeResult' % klass.__name__, 
                        (ValidRecipeResult,), 
                        self.products)

        klass.RecipeResult = a

        return klass
