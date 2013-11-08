#
# Copyright 2013 Universidad Complutense de Madrid
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

'''Simple enumeration implementation.

This implementation is based on the specification of the enum class in
Python 3.4. Some aspects are not implemented, such as methods in enum classes.

'''

from inspect import ismethod, isfunction

class _EnumVal(object):
    '''Base class for enumerated values.'''
    def __init__(self, key, val):
        self.name = key
        self.value = val

    def __str__(self):
        return "%s.%s" % (self.__class__.__name__, self.name)

    def __repr__(self):
        return "<%s.%s: %s>" % (self.__class__.__name__, self.name, self.value)

class EnumType(type):
    '''Metaclass for enumerated classes.'''
    def __new__(cls, classname, parents, attributes):

        # Custom eval type
        MyEnumVal = type(classname, (_EnumVal,), {})

        members = {}
        valid = {}
        for key, val in attributes.items():
            if not key.startswith('_') and \
                not ismethod(val) and \
                not isfunction(val):
                mm = MyEnumVal(key, val)
                members[key] = mm
            else:
                valid[key] = val
        valid['__members__'] = members
        valid['__enum_val__'] = MyEnumVal
        return super(EnumType, cls).__new__(cls, classname, parents, valid)

    def __instancecheck__(self, instance):
        return isinstance(instance, self.__enum_val__)

    def __call__(self, idx):
        for en in self.__members__.itervalues():
            if en.value == idx:
                return en
        else:
            raise ValueError('No member with value %s' % idx)

    def __getitem__(self, name):
        return self.__members__[name]

    def __getattr__(self, name):
        return self.__members__[name]

    def __iter__(self):
        return self.__members__.itervalues()

    def __contains__(self, item):
        return isinstance(item, self.__enum_val__)

class Enum(object):
    '''Base class for enumerated classes.'''
    __metaclass__ = EnumType

