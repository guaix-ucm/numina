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

import collections

import six

from numina.types.obsresult import ObservationResultType
from numina.types.obsresult import InstrumentConfigurationType
from numina.types.datatype import PlainPythonType, ListOfType
from .validator import as_list as deco_as_list
from .dataholders import EntryHolder
from .query import Ignore


class Requirement(EntryHolder):
    """Requirement holder for RecipeInput.

    Parameters
    ----------
    rtype : :class:`~numina.types.datatype.DataType` or Type[DataType]
       Object or class repressenting the yype of the requirement,
       it must be a subclass of DataType
    description : str
       Description of the Requirement. The value is used
       by `numina show-recipes` to provide human-readable documentation.
    destination : str, optional
       Name of the field in the RecipeInput object. Overrides the value
       provided by the name of the Requirement variable
    optional : bool, optional
       If `False`, the builder of the RecipeInput must provide a value
       for this Parameter. If `True` (default), the builder can skip
       this Parameter and then the default in `value` is used.
    default : optional
        The value provided by the Requirement if the RecipeInput builder
        does not provide one.
    choices : list of values, optional
        The possible values of the inputs. Any other value will raise
        an exception
    """

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

    def query_options(self):
        return self.query_opts

    def query(self, dal, obsres, options=None):
        # query opts
        if options is not None:
            # Starting with True/False
            perform_query = options
            if not perform_query:
                # we do not perform any query
                return self.default

        if isinstance(self.query_opts, Ignore):
            # we do not perform any query
            return self.default

        if self.dest is None:
            raise ValueError("destination value is not set, "
                             "use the constructor to set destination='value' "
                             "explicitly")

        val = self.type.query(self.dest, dal, obsres, options=self.query_opts)
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

    def query_constraints(self):
        return self.type.query_constraints()


def _process_nelem(nlem):
    if nlem is None:
        return False, (None, None)
    if isinstance(nlem, int):
        return True, (nlem, nlem)
    if isinstance(nlem, six.string_types):
        if nlem == '*':
            return True, (0, None)
        if nlem == '+':
            return True, (1, None)

    raise ValueError('value {} is invalid'.format(nlem))


def _recursive_type(value, nmin=None, nmax=None, accept_scalar=True):
    if isinstance(value, (list, tuple)):
        # Continue with contents of list
        if len(value) == 0:
            next_ = None
        else:
            next_ = value[0]
        final = _recursive_type(next_, accept_scalar=accept_scalar)
        return ListOfType(final, nmin=nmin, nmax=nmax, accept_scalar=accept_scalar)
    elif isinstance(value, (bool, str, int, float, complex)):
        next_ = value
        return PlainPythonType(next_)
    else:
        return value


class Parameter(Requirement):
    """The Recipe requires a plain Python type.

    Parameters
    ----------
    value : plain python type
       Default value of the parameter, the requested type is inferred from
       the type of value.
    description: str
       Description of the parameter. The value is used
       by `numina show-recipes` to provide human-readible documentation.
    destination: str, optional
       Name of the field in the RecipeInput object. Overrides the value
       provided by the name of the Parameter variable
    optional: bool, optional
       If `False`, the builder of the RecipeInput must provide a value
       for this Parameter. If `True` (default), the builder can skip
       this Parameter and then the default in `value` is used.
    choices: list of  plain python type, optional
        The possible values of the inputs. Any other value will raise
        an exception
    validator: callable, optional
        A custom validator for inputs
    accept_scalar: bool, optional
        If `True`, when `value` is a list, scalar value inputs are converted
        to list. If `False` (default), scalar values will raise an exception
        if `value` is a list
    as_list: bool, optional:
        If `True`, consider the internal type a list even if `value` is scalar
        Default is `False`
    nelem: str or int, optional:
        If nelem is '*', the list can contain any number of objects. If is '+',
        the list must contain at least 1 element. With a number, the list must
        contain that number of elements.
    """
    def __init__(self, value, description, destination=None, optional=True,
                 choices=None, validation=True, validator=None,
                 accept_scalar=False,
                 as_list=False, nelem=None,
                 ):

        if nelem is not None:
            decl_list, (nmin, nmax) = _process_nelem(nelem)
        elif as_list:
            decl_list = True
            nmin = nmax = None
        else:
            decl_list = False
            nmin = nmax = None

        is_scalar = not isinstance(value, collections.Iterable)

        if is_scalar and decl_list:
            accept_scalar = True
            value = [value]

        if isinstance(value, (bool, str, int, float, complex, list)):
            default = value
        else:
            default = None

        if validator is None:
            self.custom_validator = None
        elif callable(validator):
            if as_list:
                self.custom_validator = deco_as_list(validator)
            else:
                self.custom_validator = validator
        else:
            raise TypeError('validator must be callable or None')

        mtype = _recursive_type(value,
                                nmin=nmin, nmax=nmax,
                                accept_scalar=accept_scalar
                                )

        super(Parameter, self).__init__(
            mtype, description, destination=destination,
            optional=optional, default=default,
            choices=choices, validation=validation
            )

    def convert(self, val):
        """Convert input values to type values."""
        pre = self.type.convert(val)

        if self.custom_validator is not None:
            post = self.custom_validator(pre)
        else:
            post = pre
        return post

    def validate(self, val):
        """Validate values according to the requirement"""
        if self.validation:
            self.type.validate(val)

            if self.custom_validator is not None:
                self.custom_validator(val)

        return True


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
