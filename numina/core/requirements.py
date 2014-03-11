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

from .types import NullType, PlainPythonType
from .products import ObservationResultType
from .products import InstrumentConfigurationType
from .products import DataProduct

class Requirement(object):
    '''Requirements of Recipes
    
        :param optional: Make the Requirement optional
    
    '''
    def __init__(self, type_=None, description='', validate=False,
                dest=None, optional=False, default=None, choices=None):
        if type_ is None:
            self.type = NullType()
        elif inspect.isclass(type_):
            self.type = type_()
        else:
            self.type = type_

        self.validate = validate
        self.description = description
        self.optional = optional
        self.dest = dest
        self.default = default
        self.choices = choices
        self.hidden = False

    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s', default=%s, optional=%s, type=%s, choices=%r)" % (sclass, 
            self.dest, self.description, self.default, self.optional, self.type, self.choices)

class Parameter(Requirement):
    def __init__(self, value, description, optional=False,
                 dest=None, choices=None):
        default = value
        type_ = PlainPythonType(ref=value)

        super(Parameter, self).__init__(type_, description, 
            default=default, optional=optional, dest=dest, choices=choices)
    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s', default=%s, optional=%s, type=%s, choices=%r)" % (sclass, 
            self.dest, self.description, self.default, self.optional, 
            self.type.python_type, self.choices)
        
class DataProductRequirement(Requirement):
    '''The Recipe requires a data product of another recipe.'''
    def __init__(self, type_, description, default=None, validate=False, optional=False, dest=None, hidden=False):
        
        if inspect.isclass(type_):
            cls = type_
        else:
            cls = type_.__class__

        if not issubclass(cls, DataProduct):
            raise TypeError('%s type must derive from DataProduct' % cls)
        
        super(DataProductRequirement, self).__init__(type_, description, 
            default=default, optional=optional, dest=dest, validate=validate)

class ObservationResultRequirement(Requirement):
    '''The Recipe requires the result of an observation.'''
    def __init__(self):
        
        super(ObservationResultRequirement, self).__init__(
            ObservationResultType, "Observation Result", 
            validate=True)

    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s')" % (sclass, self.dest, self.description)

class InstrumentConfigurationRequirement(Requirement):
    '''The Recipe requires the configuration of the instrument.'''
    def __init__(self):
        
        super(InstrumentConfigurationRequirement, self).__init__(InstrumentConfigurationType, 
            "Instrument Configuration", validate=True)

    def __repr__(self):
        sclass = type(self).__name__
        return "%s(dest=%r, description='%s')" % (sclass, self.dest, self.description)

