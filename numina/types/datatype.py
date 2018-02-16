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

import inspect

from numina.exceptions import ValidationError
from numina.exceptions import NoResultFound

from .base import DataTypeBase
from .typedialect import dialect_info


class DataType(DataTypeBase):
    """Base class for input/output types of recipes.

    """
    def __init__(self, ptype, default=None, **kwds):
        super(DataType, self).__init__(**kwds)
        self.internal_type = ptype
        self.internal_dialect = dialect_info(self)
        self.internal_default = default

    def convert(self, obj):
        """Basic conversion to internal type

        This method is intended to be redefined by subclasses
        """
        return obj

    def convert_in(self, obj):
        """Basic conversion to internal type of inputs.

        This method is intended to be redefined by subclasses
        """
        return self.convert(obj)

    def convert_out(self, obj):
        """Basic conversion to internal type of outputs.

        This method is intended to be redefined by subclasses
        """
        return self.convert(obj)

    def validate(self, obj):
        """Validate convertibility to internal representation

        Returns
        -------
        bool
          True if 'obj' matches the data type

        Raises
        -------
        ValidationError
            If the validation fails

        """
        if not isinstance(obj, self.internal_type):
            raise ValidationError(obj, self.internal_type)
        return True

    def add_dialect_info(self, dialect, tipo):
        key = self.__module__ + '.' + self.__class__.__name__
        result = {'fqn': key, 'python': self.internal_type, 'type': tipo}
        self.internal_dialect[dialect] = result
        return result

    @classmethod
    def isconfiguration(cls):
        return False

    def potential_tags(self):
        return {}

    def descriptive_name(self):
        return self.name()


class AutoDataType(DataType):
    """Data type for types that are its own python type"""
    def __init__(self):
        super(AutoDataType, self).__init__(ptype=self.__class__)


class NullType(DataType):
    """Data type for None."""
    def __init__(self):
        super(NullType, self).__init__(type(None))

    def convert(self, obj):
        return None

    def validate(self, obj):
        if obj is not None:
            raise ValidationError
        return True


class PlainPythonType(DataType):
    """Data type for Python basic types."""
    def __init__(self, ref=None, validator=None):
        stype = type(ref)
        default = stype()

        if validator is None:
            self.custom_validator = None
        elif callable(validator):
            self.custom_validator = validator
        else:
            raise TypeError('validator must be callable or None')

        super(PlainPythonType, self).__init__(stype, default=default)

    def query(self, name, dal, ob, options=True):

        try:
            return self.query_on_ob(name, ob)
        except NoResultFound:
            pass

        #param = dal.search_param_req_tags(req, ob.instrument,
        #                                      ob.mode, ob.tags, ob.pipeline)
        param = dal.search_parameter(name, self, ob)
        return param.content

    def convert(self, obj):
        pre = self.internal_type(obj)

        if self.custom_validator is not None:
            m = self.custom_validator(pre)
            return m

        return pre


class ListOfType(DataType):
    """Data type for lists of other types."""
    def __init__(self, ref, index=0):
        stype = list
        if inspect.isclass(ref):
            self.internal = ref()
        else:
            self.internal = ref
        super(ListOfType, self).__init__(stype)
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
