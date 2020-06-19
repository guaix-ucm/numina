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

@singledispatch
def load(tag, obj):
    """

    Parameters
    ----------
    tag
    obj

    Returns
    -------

    """

    if hasattr(tag, '__numina_load__'):
        return tag.__numina_load__(obj)

    if hasattr(tag, '_datatype_load'):
        return tag._datatype_load(obj)

    return obj

