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

class DataType(object):
    
    def __init__(self, ptype, default=None):
        self.python_type = ptype
        self.default = default
        self.dialect = {}

    def store(self, obj):
        return obj
    
    def validate(self, obj):
        return isinstance(obj, self.python_type)

class NullType(DataType):

    def __init__(self):
        super(NullType, self).__init__(type(None))
        self.dialect = None

    def store(self, obj):
        return None

    def validate(self, obj):
        return obj is None

class PlainPythonType(DataType):
    def __init__(self, ref=None):
        stype = type(ref)
        default = stype()
        super(PlainPythonType, self).__init__(stype, default=default)
        self.dialect = stype

class ListOf(DataType):
    def __init__(self, ref):
        stype = list
        if inspect.isclass(ref):
            self.internal = ref()
        else:
            self.internal = ref
        super(ListOf, self).__init__(stype)
        self.dialect = {}

    def store(self, obj):
        result = [self.internal.store(o) for o in obj]
        return result

    def validate(self, obj):
        for o in obj:
            if not self.internal.validate(o):
                return False


import pkgutil
import importlib
import numina.ext as namespace

def default_dialect_info(obj):
    return {}

for imp, name, _is_pkg in pkgutil.walk_packages(namespace.__path__, namespace.__name__ + '.'):
    try:
        loader = imp.find_module(name)
        mod = loader.load_module(name)
        dialect_info = getattr(mod, 'dialect_info')
        if dialect_info:
            break
    except StandardError as error:
        #print name, type(error), error
        pass
else:
    dialect_info = default_dialect_info

