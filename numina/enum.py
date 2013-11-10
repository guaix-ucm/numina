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

class EnumType(type):
    '''Metaclass for enumerated classes.'''
    def __new__(cls, classname, parents, attributes):
        members = {}
        valid = {}
        for key, val in attributes.items():
            if not key.startswith('_') and \
                not ismethod(val) and \
                not ismethod(val) and \
                not isinstance(val, classmethod) and \
                not isfunction(val):
                members[key] = val
            else:
                valid[key] = val
        valid['__members__'] = members
        return super(EnumType, cls).__new__(cls, classname, parents, valid)

    def __init__(cls, classname, parents, attributes):
        for i,v in cls.__members__.items():
            m = cls.__new__(cls)
            m.__init__(i, v)
            setattr(cls, i, m)
            cls.__members__[i] = m

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

    def __repr__(self):
        return "<enum %r>" % (self.__name__)

class Enum(object):
    '''Base class for enumerated classes.'''
    __metaclass__ = EnumType

    def __init__(self, name, value):
        self.name = name
        self.value = value
        super(Enum, self).__init__()

    def __eq__(self, other):
        return (isinstance(other, self.__class__) or isinstance(self, other.__class__)) and (other.name == self.name) and (other.value == self.value)

    def __le__(self, other):
        raise TypeError('operation <= not defined for %r' % self.__class__)

    def __lt__(self, other):
        raise TypeError('operation < not defined for %r' % self.__class__)

    def __ge__(self, other):
        raise TypeError('operation >= not defined for %r' % self.__class__)

    def __gt__(self, other):
        raise TypeError('operation > not defined for %r' % self.__class__)

    def __str__(self):
        return "%s.%s" % (self.__class__.__name__, self.name)

    def __repr__(self):
        return "<%s.%s: %s>" % (self.__class__.__name__, self.name, self.value)

