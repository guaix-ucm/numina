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
Product holders
'''

import inspect

from dataproduct import DataProduct

class Product(object):
    '''Product holder for RecipeResult.'''
    def __init__(self, product_type, optional=False, *args, **kwds):

        if inspect.isclass(product_type):
            product_type = product_type()

        if isinstance(product_type, Optional):
            self.product_type = product_type.product_type
            self.optional = True
        elif isinstance(product_type, DataProduct):
            self.product_type = product_type
            self.optional = optional
        else:
            raise TypeError('product_type must be of class DataProduct')

    def __repr__(self):
        return 'Product(type=%r)' % self.product_type.__class__.__name__


class Optional(object):
    def __init__(self, product_type):

        if inspect.isclass(product_type):
            product_type = product_type()

        if isinstance(product_type, DataProduct):
            self.product_type = product_type
        else:
            raise TypeError('product_type must be of class DataProduct')

