#
# Copyright 2008-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Module to check and activate GTC compatibility code"""

_ignore_gtc_check = False

try:
    import DF
    _run_in_gtc = True
except ImportError:
    import types
    import enum

    DF = types.ModuleType('DF')

    class AdaptedOrbTypes(enum.Enum):
        TYPE_FRAME = 1
        TYPE_STRUCT = 2
        TYPE_FRAME_LIST = 3
        TYPE_STRUCT_LIST = 4
        TYPE_DOUBLE_ARRAY2D = 5

    DF.TYPE_FRAME = AdaptedOrbTypes.TYPE_FRAME
    DF.TYPE_STRUCT = AdaptedOrbTypes.TYPE_STRUCT
    DF.TYPE_FRAME_LIST = AdaptedOrbTypes.TYPE_FRAME_LIST
    DF.TYPE_STRUCT_LIST = AdaptedOrbTypes.TYPE_STRUCT_LIST
    DF.TYPE_DOUBLE_ARRAY2D = AdaptedOrbTypes.TYPE_DOUBLE_ARRAY2D

    _run_in_gtc = False


def ignore_gtc_check(value=True):
    global _ignore_gtc_check
    _ignore_gtc_check = value


def check_gtc():
    if _ignore_gtc_check:
        return False
    else:
        return _run_in_gtc
