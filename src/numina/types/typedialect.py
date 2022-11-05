#
# Copyright 2008-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


def default_dialect_info(obj):
    key = obj.__module__ + '.' + obj.__class__.__name__
    result = {'base': {'fqn': key, 'python': obj.internal_type}}
    return result


dialect_info = default_dialect_info
