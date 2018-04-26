#
# Copyright 2010-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import warnings

try:
    from functools import singledispatch
except ImportError:
    from pkgutil import simplegeneric as singledispatch

import numpy

from numina.types.dataframe import DataFrame
from numina.types.frame import dump_dataframe
from numina.types.array import dump_numpy_array


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