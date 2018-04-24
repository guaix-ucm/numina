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

@singledispatch
def load(tag, obj):

    if hasattr(tag, '__numina_load__'):
        msg = "Usage of '__numina_load__' is deprecated, use '_datatype_load' instead"
        warnings.warn(msg, DeprecationWarning)
        return tag.__numina_load__(obj)

    if hasattr(tag, '_datatype_load'):
        return tag._datatype_load(obj)

    return obj

