#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Utilities for DAL"""


def tags_are_valid(subset, superset):
    """Validate tags"""
    for key, val in subset.items():
        if key in superset and superset[key] != val:
            return False
    return True
