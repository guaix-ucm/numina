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

from __future__ import print_function

import inspect

from numina.exceptions import Error
from .products import DataProduct, ValidationError
from .oresult import ObservationResult
from .pipeline import InstrumentConfiguration

class RequirementError(Error):
    '''Error in the parameters of a recipe.'''
    def __init__(self, txt):
        super(RequirementError, self).__init__(txt)

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

def type_validate(tipo, valor):
    '''Basic validation.'''
    if hasattr(tipo, 'validate'):
        tipo.validate(valor)

    else:
        if not isinstance(valor, tipo.__class__):
            raise ValidationError("%r is not an instance of %r" % (valor, tipo))

class RequirementParser(object):
    '''RecipeRequirement builder.'''
    def __init__(self, recipe, lookupclass=RequirementLookup):
        if not inspect.isclass(recipe):
            recipe = recipe.__class__
        self.requirements = recipe.__requires__
        self.rClass = recipe.RecipeRequirements
        self.lc = lookupclass()

    def parse(self, metadata, validate=False):
        '''Build the RecipeRequirement object from available metadata.'''
        parameters = {}
        
        for req in self.requirements:
            if req.dest is None:
                # FIXME: add warning or something here
                continue
            value = self.lc.lookup(req, metadata)
            if req.choices and (value not in req.choiches):
                raise RequirementError('%s not in %s' % (value, req.choices))

            # Build value
            if hasattr(req.type, 'store'):
                mm = req.type.store(value)
            else:
                mm = value
            # validate
            if req.validate or validate:
                if mm is not None or not req.optional:
                    type_validate(req.type, mm)
                
            parameters[req.dest] = mm
        names = self.rClass(**parameters)

        return names

    def print_requirements(self, pad=''):
        
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
        
            print("%s%s [%s]" % (pad, dispname, req.description))

class Requirement(object):
    '''Requirements of Recipes
    
        :param optional: Make the Requirement optional
    
    '''
    def __init__(self, description, value=None, optional=False, 
                 validate=False, type=None,
                 dest=None, hidden=False, choices=None):
        self.default = value
        self.description = description
        self.optional = optional
        self.validate = validate

        if type is None:
            self.type = DataProduct()
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
        
class ObservationResultRequirement(Requirement):
    '''The Recipe requires the result of an observation.'''
    def __init__(self):
        
        super(ObservationResultRequirement, self).__init__("Observation Result", 
            type=ObservationResult, validate=True)

    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s')" % (sclass, self.dest, self.description)

class InstrumentConfigurationType(object):
    '''The type of InstrumentConfiguration.'''
    def validate(self, value):
        if not isinstance(value, InstrumentConfiguration):
            raise ValidationError('%r is not an instance of InstrumentConfiguration')

class InstrumentConfigurationRequirement(Requirement):
    '''The Recipe requires the configuration of the instrument.'''
    def __init__(self):
        
        super(InstrumentConfigurationRequirement, self).__init__("Instrument Configuration", 
            type=InstrumentConfigurationType, validate=True)

    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s')" % (sclass, self.dest, self.description)

class DataProductRequirement(Requirement):
    '''The Recipe requires a data product of another recipe.'''
    def __init__(self, valueclass, description, optional=False, dest=None, hidden=False):
        
        super(DataProductRequirement, self).__init__(description, optional=optional, 
                                                     type=valueclass, dest=dest, hidden=hidden)
        if not inspect.isclass(valueclass):
            valueclass = valueclass.__class__
             
        if not isinstance(self.type, DataProduct):
            raise TypeError('valueclass must derive from DataProduct')
        
