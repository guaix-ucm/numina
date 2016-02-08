#
# Copyright 2010-2015 Universidad Complutense de Madrid
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

import warnings

try:
    from functools import singledispatch
except ImportError:
    from pkgutil import simplegeneric as singledispatch

import numpy

from numina.core.dataframe import DataFrame
from numina.core.products import dump_dataframe,dump_numpy_array

@singledispatch
def dump(tag, obj, where):

    if hasattr(tag, '__numina_dump__'):
        msg = "Usage of '__numina_dump__' is deprecated, use '_datatype_dump' instead"
        warnings.warn(msg, DeprecationWarning)
        return tag.__numina_dump__(obj, where)

    if hasattr(tag, '_datatype_dump'):
        return tag._datatype_dump(obj, where)

    return obj

# It's not clear if I need to register these three
# functions


@dump.register(list)
def _(tag, obj, where):
    return [dump(tag, o, where) for o in obj]


dump.register(numpy.ndarray, dump_numpy_array)


dump.register(DataFrame, dump_dataframe)