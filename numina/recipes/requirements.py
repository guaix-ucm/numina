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

from numina.exceptions import Error
from .products import DataProduct

class RequirementError(Error):
    '''Error in the parameters of a recipe.'''
    def __init__(self, txt):
        super(RequirementError, self).__init__(txt)

class RequirementLookup(object):
    def lookup(self, req, source):
        if req.name in source:
            return source[req.name]
        elif req.optional:
            return None
        elif req.default is not None:
            return req.default
        else:
            raise RequirementError('Requirement %s must be defined' % req.name)

class RequirementParser(object):
    
    def __init__(self, requirements, lookupclass=RequirementLookup):
        self.requirements = requirements
        self.lc = lookupclass()

    def parse(self, metadata):
        parameters = {}
        
        for req in self.requirements:
            value = self.lc.lookup(req, metadata)
            if req.choices and (value not in req.choices):
                raise RequirementError('%s not in %s' % (value, req.choices))
                
            parameters[req.dest]= value 
        return parameters

    def print_requirements(self):
        
        for req in self.requirements:
            dispname = req.name
    
            if req.optional:
                dispname = req.name + '(optional)'
    
            if req.default is not None:
                dispname = dispname + '=' + str(req.default)
        
            print "%s [%s]" % (dispname, req.description)

class Requirement(object):
    '''Requirements of Recipes
    
        :param optional: Make the Requirement optional
    
    '''
    def __init__(self, name, description, value=None, optional=False, type=None,
                 dest=None, choices=None):
        self.name = name
        self.default = value
        self.description = description
        self.optional = optional
        self.type = type
        self.choices = choices
        
        if dest is None:
            self.dest = name
        
    def __repr__(self):
        sclass = type(self).__name__
        return "%s(name='%s', description='%s', default=%s, optional=%s, type=%s, dest='%s', choices=%r)" % (sclass, 
            self.name, self.description, self.default, self.optional, self.type, self.dest, self.choices)
        

        
class Parameter(Requirement):
    def __init__(self, name, value, description, optional=False, type=None, choices=None):
        super(Parameter, self).__init__(name, description, 
            value=value, optional=optional, type=type, choices=choices)
        
class DataProductRequirement(Requirement):
    def __init__(self, name, valueclass, description, optional=False):
        
        if not inspect.isclass(valueclass):
            valueclass = valueclass.__class__
             
        if not issubclass(valueclass, DataProduct):
            raise TypeError('valueclass must derive from DataProduct')
        
        super(DataProductRequirement, self).__init__(name, description, optional=optional,
                                                   type=valueclass)
