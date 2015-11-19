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

import inspect

from numina.exceptions import ValidationError
from .typedialect import dialect_info


class DataType(object):

    def __init__(self, ptype, default=None):
        self.python_type = ptype
        self.default = default
        self.dialect = dialect_info(self)

    def convert(self, obj):
        return obj

    def validate(self, obj):
        if not isinstance(obj, self.python_type):
            raise ValidationError(obj, self.python_type)
        return True

    def __numina_dump__(self, obj, where):
        return obj

    def __repr__(self):
        sclass = type(self).__name__
        return "%s()" % (sclass, )


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
    def __init__(self, ref):
        stype = list
        if inspect.isclass(ref):
            self.internal = ref()
        else:
            self.internal = ref
        super(ListOfType, self).__init__(stype)
        self.dialect = {}

    def convert(self, obj):
        result = [self.internal.convert(o) for o in obj]
        return result

    def validate(self, obj):
        for o in obj:
            self.internal.validate(o)
        return True