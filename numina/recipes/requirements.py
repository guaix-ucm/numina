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

from numina.exceptions import ParameterError
from .products import DataProduct

class RequirementParser(object):
    
    def __init__(self, requirements, lookup):
        self.requirements = requirements
        self.lookup = lookup

    def parse(self, metadata):
        parameters = {}
        
        for req in self.requirements:
            value = self.lookup(req, metadata)
            if req.choiches and (value not in req.choiches):
                raise ParameterError('%s not in %s' % (value, req.choices))
                
            parameters[req.dest]= value 
        return parameters

class Requirement(object):
    '''Requirements of Recipes
    
        :param optional: Make the Requirement optional
    
    '''
    def __init__(self, name, description, value=None, optional=False, type=None,
                 dest=None):
        self.name = name
        self.default = value
        self.description = description
        self.optional = optional
        self.type = type
        
        if dest is None:
            self.dest = name
        
    def __repr__(self):
        sclass = type(self).__name__
        return "%s(name='%s', description='%s', default=%s, optional=%s, type=%s, dest='%s')" % (sclass, 
            self.name, self.description, self.default, self.optional, self.type, self.dest)
        
class Parameter(Requirement):
    def __init__(self, name, value, description, optional=False, type=None):
        super(Parameter, self).__init__(name, description, 
            value=value, optional=optional, type=type)
        
class DataProductRequirement(Requirement):
    def __init__(self, name, valueclass, description, optional=False):
        
        if not inspect.isclass(valueclass):
            valueclass = valueclass.__class__
             
        if not issubclass(valueclass, DataProduct):
            raise TypeError('valueclass must derive from DataProduct')
        
        super(DataProductRequirement, self).__init__(name, description, optional=optional,
                                                   type=valueclass)
