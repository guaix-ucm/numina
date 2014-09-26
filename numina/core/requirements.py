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
Recipe requirement holders
'''

from .products import DataProductType
from .products import ObservationResultType
from .products import InstrumentConfigurationType

from .dataholders import EntryHolder


class Requirement(EntryHolder):
    '''Requirement holder holder for RecipeRequirement.'''
    def __init__(self, rtype, description, validation=True,
                 dest=None, optional=False, default=None, choices=None):
        super(Requirement, self).__init__(
            rtype, description, dest,
            optional, default, choices=choices,
            validation=validation
            )

        self.hidden = False

    def __repr__(self):
        sclass = type(self).__name__
        fmt = ("%s(dest=%r, description='%s', "
               "default=%s, optional=%s, type=%s, choices=%r)"
               )
        return fmt % (sclass, self.dest, self.description, self.default,
                      self.optional, self.type, self.choices)


class Parameter(Requirement):
    '''The Recipe requires a plain Python type.'''
    def __init__(self, value, description, dest=None, optional=False,
                 choices=None, validation=True):
        rtype = type(value)
        super(Parameter, self).__init__(
            rtype, description, dest=dest,
            optional=optional, default=value,
            choices=choices, validation=validation
            )


class DataProductRequirement(Requirement):
    '''The Recipe requires a data product of another recipe.'''
    def __init__(self, rtype, description, validation=True,
                 dest=None, optional=False, default=None):
        super(DataProductRequirement, self).__init__(
            rtype, description, dest=dest, optional=optional,
            default=default, validation=validation
            )

        if not isinstance(self.type, DataProductType):
            raise TypeError(
                '%s type must derive from DataProduct' % self.type
                )


class ObservationResultRequirement(Requirement):
    '''The Recipe requires the result of an observation.'''
    def __init__(self):

        super(ObservationResultRequirement, self).__init__(
            ObservationResultType, "Observation Result"
            )

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(dest=%r, description='%s')"
        msg = fmt % (sclass, self.dest, self.description)
        return msg


class InstrumentConfigurationRequirement(Requirement):
    '''The Recipe requires the configuration of the instrument.'''
    def __init__(self):

        super(InstrumentConfigurationRequirement, self).__init__(
            InstrumentConfigurationType,
            "Instrument Configuration",
            validation=False
            )

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(dest=%r, description='%s')"
        msg = fmt % (sclass, self.dest, self.description)
        return msg
