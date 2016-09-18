#
# Copyright 2008-2016 Universidad Complutense de Madrid
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

import inspect

from numina.exceptions import ValidationError
from .typedialect import dialect_info


class DataType(object):

    def __init__(self, ptype, default=None):
        self.internal_type = ptype
        self.internal_dialect = dialect_info(self)
        self.internal_default = default

    def convert(self, obj):
        return obj

    def convert_in(self, obj):
        return self.convert(obj)

    def convert_out(self, obj):
        return self.convert(obj)

    def validate(self, obj):
        if not isinstance(obj, self.internal_type):
            raise ValidationError(obj, self.internal_type)
        return True

    def _datatype_dump(self, obj, where):
        return obj

    def _datatype_load(self, obj):
        return obj

    @classmethod
    def isproduct(cls):
        return False

    def __repr__(self):
        sclass = type(self).__name__
        return "%s()" % (sclass, )


class AutoDataType(DataType):
    """Data type for types that are its own python type"""
    def __init__(self):
        super(AutoDataType, self).__init__(ptype=self.__class__)


class NullType(DataType):

    def __init__(self):
        super(NullType, self).__init__(type(None))
        self.dialect = None

    def convert(self, obj):
        return None

    def validate(self, obj):
        if obj is not None:
            raise ValidationError
        return True


class PlainPythonType(DataType):
    def __init__(self, ref=None):
        stype = type(ref)
        default = stype()
        super(PlainPythonType, self).__init__(stype, default=default)
        self.dialect = stype


class ListOfType(DataType):
    def __init__(self, ref, index=0):
        stype = list
        if inspect.isclass(ref):
            self.internal = ref()
        else:
            self.internal = ref
        super(ListOfType, self).__init__(stype)
        self.dialect = {}
        self.index = index

    def convert(self, obj):
        result = [self.internal.convert(o) for o in obj]
        return result

    def validate(self, obj):
        for o in obj:
            self.internal.validate(o)
        return True

    def _datatype_dump(self, objs, where):
        result = []
        old_dest = where.destination
        for idx, obj in enumerate(objs, start=self.index):
            where.destination = old_dest + str(idx)
            res = self.internal._datatype_dump(obj, where)
            result.append(res)
        where.destination = old_dest
        return result
