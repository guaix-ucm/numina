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
def load(tag, obj):
    """Load a value of the type `tag` from its serialized form `obj`.

    This is how numina loads all the values of the requirements, from
    the DAL or from the observation result. The type defines how with
    the method ``_datatype_load(obj)``; ``__numina_load__(obj)`` is also
    supported, and used first. If it defines none, `obj` is returned.

    If the type declares the formats it reads (``__numina_formats__``) and
    `obj` is a file, the type and the format of the envelope of the
    file are checked, and the file is read with the method of its
    format, see :func:`numina.store.envelope.load`.

    Parameters
    ----------
    tag : DataType
        Type of the value.
    obj
        Serialized value, as the name of a file.

    Returns
    -------
    The value.
    """

    if getattr(tag, "__numina_formats__", None) is not None:
        from . import envelope

        if envelope.is_file(obj):
            return envelope.load(tag, obj)

    if hasattr(tag, "__numina_load__"):
        return tag.__numina_load__(obj)

    if hasattr(tag, "_datatype_load"):
        return tag._datatype_load(obj)

    return obj
