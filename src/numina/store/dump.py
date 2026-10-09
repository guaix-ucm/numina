#
# Copyright 2010-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


try:
    from functools import singledispatch
except ImportError:
    from pkgutil import simplegeneric as singledispatch


@singledispatch
def dump(tag, obj, where):
    """Save a value of the type `tag`, return its serialized form.

    This is how numina saves the values of the results of the recipes.
    The type defines how with the method ``_datatype_dump(obj, where)``;
    ``__numina_dump__(obj, where)`` is also supported, and used first.
    If it defines none, `obj` is returned.

    Parameters
    ----------
    tag : DataType
        Type of the value.
    obj
        The value.
    where
        Base of the name of the file, if the value is saved in a file.

    Returns
    -------
    The serialized value, as the name of the file.
    """

    if hasattr(tag, "__numina_dump__"):
        return tag.__numina_dump__(obj, where)

    if hasattr(tag, "_datatype_dump"):
        return tag._datatype_dump(obj, where)

    return obj
