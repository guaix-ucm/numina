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

import inspect

from .products import DataProduct


class Requirement(object):
    '''Requirements of Recipes
    
        :param optional: Make the Requirement optional
    
    '''
    def __init__(self, name, value, description, optional=False):
        self.name = name
        self.value = value
        self.description = description
        self.optional = optional
        
    def lookup(self, params):    
        if self.name in params:
            # FIXME: add validation
            return params[self.name]
        elif self.optional:
            return None
        else:
            return self.value

class Parameter(Requirement):
    def __init__(self, name, value, description, optional=False):
        super(Parameter, self).__init__(name, value, 
description, optional=optional)
        
        
        
class DataProductParameter(Parameter):
    def __init__(self, name, valueclass, description, optional=False):
        
        self.default = None
        
        if not inspect.isclass(valueclass):
            self.default = valueclass
            valueclass = valueclass.__class__
             
        if not issubclass(valueclass, DataProduct):
            raise TypeError('valueclass must derive from DataProduct')
        
        super(DataProductParameter, self).__init__(name, valueclass, 
                                                   description, optional)

    def lookup(self, params):    
        if self.name in params:
            # FIXME: add validation
            return params[self.name]
        elif self.soft:
            return None
        elif self.default is not None:
            return self.default
        else:
            raise LookupError('parameter %s must be defined' % self.name)
