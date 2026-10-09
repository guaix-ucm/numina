#
# Copyright 2008-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Description of the data types for other systems (dialects)"""


def default_dialect_info(obj):
    """Return the information of the base dialect of a data type.

    It is a dictionary with the key 'base', whose value contains the fully
    qualified name of the class of the type ('fqn') and its Python type
    ('python'). Other dialects, as 'gtc', are added with
    :meth:`numina.types.datatype.DataType.add_dialect_info`.
    """
    key = obj.__module__ + "." + obj.__class__.__name__
    result = {"base": {"fqn": key, "python": obj.internal_type}}
    return result


dialect_info = default_dialect_info
