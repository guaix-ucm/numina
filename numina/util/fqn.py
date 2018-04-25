#
# Copyright 2011-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Import objects by name"""

import inspect


def fully_qualified_name(obj, sep='.'):
    """Return fully qualified name from object"""
    if inspect.isclass(obj):
        return obj.__module__ + sep + obj.__name__
    else:
        return obj.__module__ + sep + obj.__class__.__name__



