#
# Copyright 2010-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


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
    """

    Parameters
    ----------
    tag
    obj
    where

    Returns
    -------

    """

    if hasattr(tag, '__numina_dump__'):
        return tag.__numina_dump__(obj, where)

    if hasattr(tag, '_datatype_dump'):
        return tag._datatype_dump(obj, where)

    return obj

# It's not clear if I need to register these three
# functions


@dump.register(list)
def _(tag, obj, where):
    return [dump(tag, o, where) for o in obj]


dump.register(numpy.ndarray, lambda t, o, w: dump_numpy_array(o, w))


dump.register(DataFrame, lambda t,o,w: dump_dataframe(o, w))
