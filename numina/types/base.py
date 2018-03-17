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

from numina.util.parser import parse_arg_line
from numina.exceptions import NoResultFound
from numina.datamodel import DataModel
from numina.core.query import Result


class DataTypeBase(object):
    """Base class for input/output types of recipes.

    """
    def __init__(self, *args, **kwds):
        datamodel = kwds.get('datamodel')

        if datamodel is not None:
            if inspect.isclass(datamodel):
                self.datamodel = datamodel()
            else:
                self.datamodel = datamodel
        else:
            self.datamodel = DataModel()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

    def _datatype_dump(self, obj, where):
        return obj

    def _datatype_load(self, obj):
        return obj

    def query(self, name, dal, obsres, options=None):

        try:
            return self.query_on_ob(name, obsres)
        except NoResultFound:
            pass

        if isinstance(options, Result):
            value = dal.search_result_relative(name, self, obsres,
                                               mode=options.mode,
                                               field=options.field,
                                               node=options.node)
            return value.content

        if self.isproduct():
            # if not, the normal query
            prod = dal.search_product(name, self, obsres)
            return prod.content
        else:
            param = dal.search_parameter(name, self, obsres)
            return param.content

    def query_on_ob(self, key, ob):
        # First check if the requirement is embedded
        # in the observation result
        # It can in ob.requirements
        # or directly in the structure (as in GTC)
        if key in ob.requirements:
            content = ob.requirements[key]
            value = self._datatype_load(content)
            return value
        try:
            return getattr(ob, key)
        except AttributeError:
            raise NoResultFound("DataType.query_on_ob")

    def on_query_not_found(self, notfound):
        pass

    @classmethod
    def isproduct(cls):
        """Check if the DataType is the product of a Recipe"""
        return False

    def __repr__(self):
        sclass = type(self).__name__
        return "%s()" % (sclass, )

    def name(self):
        """Unique name of the datatype"""
        return self.__repr__()

    @classmethod
    def from_name(cls, name):
        # name is in the form Class(arg1=val1, arg2=val2)
        # find first (
        fp = name.find('(')
        sp = name.find(')')
        if (fp == -1) or (sp == -1) or (sp < fp):
            # paren not found
            klass = name
            kwargs = {}
        else:
            # parse things between parens
            klass = name[:fp]
            fargs = name[fp+1:sp]
            kwargs = parse_arg_line(fargs)

        if klass == cls.__name__:
            # create thing
            obj = cls.__new__(cls)
            obj.__init__(**kwargs)

            return obj
        else:
            raise TypeError(name)

    @staticmethod
    def create_db_info():
        """Create metadata structure"""
        result = {}
        result['instrument'] = ''
        result['uuid'] = ''
        result['tags'] = {}
        result['type'] = ''
        result['mode'] = ''
        result['observation_date'] = ""
        result['origin'] = {}
        return result

    def extract_db_info(self, obj, db_info_keys):
        """Extract metadata from serialized file"""
        result = self.create_db_info()
        return result

    def update_meta_info(self):
        """Extract my metadata"""
        result = self.create_db_info()
        return result
