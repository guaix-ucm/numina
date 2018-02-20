#
# Copyright 2014-2018 Universidad Complutense de Madrid
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

"""FITS header schema and validation.

This module is a simplification of the FITS Schema defined
by Erik Bray here:
http://embray.github.io/PyFITS/schema/users_guide/users_schema.html

If this schema implementation reaches pyfits/astropy stable,
we will use it instead of ours, with schema definitions
being the same.

"""


class SchemaValidationError(Exception):
    """Exception raised when a Schema does not validate a FITS header."""
    pass


class SchemaDefinitionError(Exception):
    """Exception raised when a FITS Schema definition is not valid."""
    pass


def _from_ipt(value):
    if isinstance(value, bool):
        return (value, bool)
    if isinstance(value, str):
        return (value, str)
    elif isinstance(value, int):
        return (value, int)
    elif isinstance(value, float):
        return (value, float)
    elif isinstance(value, complex):
        return (value, complex)
    elif value in [str, int, float, complex, bool]:
        return (None, value)
    elif isinstance(value, list):
        if value:
            _, type_ = _from_ipt(value[0])
            return value, type_
        else:
            raise SchemaDefinitionError(value)
    else:
        raise SchemaDefinitionError(value)


class SchemaKeyword(object):
    """A keyword in the schema"""
    def __init__(self, name, mandatory=False, valid=True,
                 value=None):
        self.name = name
        self.mandatory = mandatory
        self.valid = valid
        if self.mandatory and not self.valid:
            raise SchemaDefinitionError(
                "keyword 'cannot be 'mandatory' and "
                "'not valid'"
                )
        self.choose = False
        self.valcheck = False
        self.value = None
        self.type_ = None
        if value is not None:
            self.value, self.type_ = _from_ipt(value)
            if self.value is not None:
                self.valcheck = True
                if isinstance(self.value, list):
                    self.choose = True

    def validate(self, header):
        sname = 'schema'
        # check the keyword is defined
        val = header.get(self.name)

        if val is None:
            if self.mandatory:
                raise SchemaValidationError(
                    sname, 'mandatory keyword %r '
                    'missing from header' % self.name)

            # In the rest of cases
            return True
        else:
            if not self.valid:
                raise SchemaValidationError(
                    sname, 'invalid keyword %r present in header'
                    % self.name)

        # Cases here
        # val is not None and key id mandatory or valid

        if not self.type_:
            # We dont have type information
            # Nothing more to do
            return True
        else:
            if not isinstance(val, self.type_):
                raise SchemaValidationError(
                    sname, 'keyword %r is required to have a value of type %r'
                    '; got a value of type %r instead' %
                    (self.name, self.type_.__name__, type(val).__name__))
            # Check value
            if self.choose:
                if val not in self.value:
                    raise SchemaValidationError(
                        sname,
                        'keyword %r is required to have one of the values %r; '
                        'got %r instead' %
                        (self.name, self.value, val))
                else:
                    return True
            elif self.valcheck:
                if val != self.value:
                    raise SchemaValidationError(
                        sname,
                        'keyword %r is required to have the value %r; got '
                        '%r instead' % (self.name, self.value, val))
            else:
                pass

        return True


class Schema(object):
    """A FITS schema"""
    def __init__(self, sc):
        self.kwl = []
        self.extend(sc)

    def validate(self, header):
        for ll in self.kwl:
            ll.validate(header)

    def extend(self, sc):
        kw = sc['keywords']
        for k, v in kw.items():
            mandatory = v.get('mandatory', False)
            valid = v.get('valid', True)
            value = v.get('value', None)
            sk = SchemaKeyword(
                k, mandatory=mandatory,
                valid=valid, value=value
                )
            self.kwl.append(sk)
