#
# Copyright 2008-2018 Universidad Complutense de Madrid
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

"""
Recipe requirement holders
"""


from numina.types.obsresult import ObservationResultType
from numina.types.obsresult import InstrumentConfigurationType
from numina.types.datatype import PlainPythonType
from .dataholders import EntryHolder
from .query import QueryModifier, Ignore, Result


class Requirement(EntryHolder):
    """Requirement holder holder for RecipeRequirement."""
    def __init__(self, rtype, description, destination=None, optional=False,
                 default=None, choices=None, validation=True, query_opts=None):
        super(Requirement, self).__init__(
            rtype, description, destination=destination,
            optional=optional, default=default, choices=choices,
            validation=validation
            )

        self.query_opts = query_opts

        self.hidden = False

    def convert(self, val):
        return self.type.convert_in(val)

    def query(self, dal, obsres, options=None):
        # query opts
        if isinstance(self.query_opts, Ignore):
            # we do not perform any query
            return self.default_value()

        # FIX merge somehow...
        # options and self.query_options
        options = self.query_opts

        val = self.type.query(self.dest, dal, obsres, options=options)
        return val

    def on_query_not_found(self, notfound):
        self.type.on_query_not_found(notfound)

    def __repr__(self):
        sclass = type(self).__name__
        fmt = ("%s(dest=%r, description='%s', "
               "default=%s, optional=%s, type=%s, choices=%r)"
               )
        return fmt % (sclass, self.dest, self.description, self.default,
                      self.optional, self.type, self.choices)


class Parameter(Requirement):
    """The Recipe requires a plain Python type."""
    def __init__(self, value, description, destination=None, optional=False,
                 choices=None, validation=True, validator=None):
        if isinstance(value, (bool, str, int, float, complex, list)):
            optional = True
            default = value
        else:
            default = None
        rtype = type(value)
        mtype = PlainPythonType(ref=rtype(), validator=validator)

        super(Parameter, self).__init__(
            mtype, description, destination=destination,
            optional=optional, default=default,
            choices=choices, validation=validation
            )


class ObservationResultRequirement(Requirement):
    """The Recipe requires the result of an observation."""
    def __init__(self, query_opts=None):

        super(ObservationResultRequirement, self).__init__(
            ObservationResultType, "Observation Result",
            query_opts=query_opts
            )

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(dest=%r, description='%s')"
        msg = fmt % (sclass, self.dest, self.description)
        return msg


class InstrumentConfigurationRequirement(Requirement):
    """The Recipe requires the configuration of the instrument."""
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
