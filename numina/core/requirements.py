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

from __future__ import print_function

import inspect

from numina.exceptions import Error
from .products import DataProduct

class RequirementError(Error):
    '''Error in the parameters of a recipe.'''
    def __init__(self, txt):
        super(RequirementError, self).__init__(txt)

class Names(object):
    def __init__(self, **kwds):
        for name, val in kwds.iteritems():
            setattr(self, name, val)

    __hash__ = None

    def __contains__(self, key):
        return key in self.__dict__

class RequirementLookup(object):
    def lookup(self, req, source):
        if req.dest in source:
            return source[req.dest]
        elif req.optional:
            return None
        elif req.default is not None:
            return req.default
        else:
            raise RequirementError('Requirement %r must be defined' % req.dest)

class RequirementParser(object):
    
    def __init__(self, recipe, lookupclass=RequirementLookup):
        if not inspect.isclass(recipe):
            recipe = recipe.__class__
        self.requirements = recipe.__requires__
        self.rClass = recipe.RecipeRequirements
        self.lc = lookupclass()

    def parse(self, metadata):
        parameters = {}
        
        for req in self.requirements:
            if req.dest is None:
                # FIXME: add warning or something here
                continue
            value = self.lc.lookup(req, metadata)
            if req.choices and (value not in req.choiches):
                raise RequirementError('%s not in %s' % (value, req.choices))

            # Build value
            mm = req.type.store(value)
                
            parameters[req.dest] = mm
        names = self.rClass(**parameters)

        return names

    def print_requirements(self):
        
        for req in self.requirements:
            if req.dest is None:
                # FIXME: add warning or something here
                continue
            if req.hidden:
                # I Do not want to print it
                continue
            dispname = req.dest
    
            if req.optional:
                dispname = dispname + '(optional)'
    
            if req.default is not None:
                dispname = dispname + '=' + str(req.default)
        
            print("%s [%s]" % (dispname, req.description))

class _RequirementParser(object):
    
    def __init__(self, requirements, lookupclass=RequirementLookup):
        self.requirements = requirements
        self.lc = lookupclass()

    def parse(self, metadata):
        parameters = {}
        
        for req in self.requirements:
            if req.dest is None:
                # FIXME: add warning or something here
                continue
            value = self.lc.lookup(req, metadata)
            if req.choices and (value not in req.choices):
                raise RequirementError('%s not in %s' % (value, req.choices))
                
            parameters[req.dest]= value 
        return parameters

    def parse2(self, metadata):
        parameters = {}
        
        for req in self.requirements:
            if req.dest is None:
                # FIXME: add warning or something here
                continue
            value = self.lc.lookup(req, metadata)
            if req.choices and (value not in req.choiches):
                raise RequirementError('%s not in %s' % (value, req.choices))

            # Build value
            mm = req.type.store(value)
                
            parameters[req.dest] = mm
        names = Names(**parameters)

        return names

    def print_requirements(self):
        
        for req in self.requirements:
            if req.dest is None:
                # FIXME: add warning or something here
                continue
            if req.hidden:
                # I Do not want to print it
                continue
            dispname = req.dest
    
            if req.optional:
                dispname = dispname + '(optional)'
    
            if req.default is not None:
                dispname = dispname + '=' + str(req.default)
        
            print("%s [%s]" % (dispname, req.description))

class Requirement(object):
    '''Requirements of Recipes
    
        :param optional: Make the Requirement optional
    
    '''
    def __init__(self, description, value=None, optional=False, type=None,
                 dest=None, hidden=False, choices=None):
        self.default = value
        self.description = description
        self.optional = optional

        if type is None:
            self.type = None
        elif inspect.isclass(type):
            self.type = type()
        else:
            self.type = type
        self.dest = dest
        self.hidden = hidden
        self.choices = choices
        
    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s', default=%s, optional=%s, type=%s, choices=%r)" % (sclass, 
            self.dest, self.description, self.default, self.optional, self.type, self.choices)

class Parameter(Requirement):
    def __init__(self, value, description, optional=False, type=None, choices=None, 
                 dest=None, hidden=False):
        super(Parameter, self).__init__(description, 
            value=value, optional=optional, type=type, dest=dest, hidden=hidden, choices=choices)
        
class DataProductRequirement(Requirement):
    def __init__(self, valueclass, description, optional=False, dest=None, hidden=False):
        
        super(DataProductRequirement, self).__init__(description, optional=optional, 
                                                     type=valueclass, dest=dest, hidden=hidden)
        if not inspect.isclass(valueclass):
            valueclass = valueclass.__class__
             
        if not isinstance(self.type, DataProduct):
            raise TypeError('valueclass must derive from DataProduct')
        
