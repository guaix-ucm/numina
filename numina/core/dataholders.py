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
Recipe requirements
"""

import inspect

import numina.exceptions


class EntryHolder(object):
    def __init__(self, tipo, description, destination, optional,
                 default, choices=None, validation=True):

        from numina.types.datatype import NullType, PlainPythonType

        super(EntryHolder, self).__init__()

        if tipo is None:
            self.type = NullType()
        elif tipo in [bool, str, int, float, complex, list]:
            self.type = PlainPythonType(ref=tipo())
        elif inspect.isclass(tipo):
            self.type = tipo()
        else:
            self.type = tipo

        self.description = description
        self.optional = optional
        self.dest = destination
        self.default = default
        self.choices = choices
        self.validation = validation

    def __get__(self, instance, owner):
        """Getter of the descriptor protocol."""
        if instance is None:
            return self
        else:
            if self.dest not in instance._numina_desc_val:
                instance._numina_desc_val[self.dest] = self.default_value()

            return instance._numina_desc_val[self.dest]

    def __set__(self, instance, value):
        """Setter of the descriptor protocol."""
        try:
            cval = self.convert(value)
            if self.choices and (cval not in self.choices):
                errmsg = '{} not in {}'.format(cval, self.choices)
                raise numina.exceptions.ValidationError(errmsg)
        except numina.exceptions.ValidationError as err:

            if len(err.args) == 0:
                errmsg = 'UNDEFINED ERROR'
                rem = ()
            else:
                errmsg = err.args[0]
                rem = err.args[1:]

            msg = '"{}": {}'.format(self.dest, errmsg)
            newargs = (msg, ) + rem
            raise numina.exceptions.ValidationError(*newargs)

        instance._numina_desc_val[self.dest] = cval

    def convert(self, val):
        return self.type.convert(val)

    def validate(self, val):
        if self.validation:
            return self.type.validate(val)
        return True

    def default_value(self):
        if self.default is not None:
            return self.convert(self.default)
        if self.type.internal_default is not None:
            return self.type.internal_default
        if self.optional:
            return None
        else:
            fmt = 'Required {0!r} of type {1!r} is not defined'
            msg = fmt.format(self.dest, self.type)
            raise ValueError(msg)


class Result(EntryHolder):
    """Result holder for RecipeResult."""
    def __init__(self, ptype, description="", validation=True,
                 destination=None, optional=False, default=None, choices=None):
        super(Result, self).__init__(
            ptype, description, destination=destination, optional=optional,
            default=default, choices=choices, validation=validation
            )

#        if not isinstance(self.type, DataProductType):
#            raise TypeError('type must be of class DataProduct')

    def __repr__(self):
        return 'Result(type=%r, dest=%r)' % (self.type, self.dest)

    def convert(self, val):
        return self.type.convert_out(val)


class Product(Result):
    """Product holder for RecipeResult."""

    def __init__(self, ptype, description="", validation=True,
                 destination=None, optional=False, default=None, choices=None):
        super(Product, self).__init__(
            ptype,
            description=description,
            validation=validation,
            destination=destination,
            optional=optional,
            default=default,
            choices=choices
        )

        import warnings
        warnings.warn("The 'Product' class was renamed to 'Result'", DeprecationWarning)

    def __repr__(self):
        return 'Product(type=%r, dest=%r)' % (self.type, self.dest)
