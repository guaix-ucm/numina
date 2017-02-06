#
# Copyright 2008-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#


"""Module to check and activate GTC compatibility code"""

_ignore_gtc_check = False

try:
    import DF
    _run_in_gtc = True
except ImportError:
    import types
    import numina.enum
    # FIXME: workaround
    DF = types.ModuleType('DF')

    class AdaptedOrbTypes(numina.enum.Enum):
        TYPE_FRAME = 1
        TYPE_STRUCT = 2
        TYPE_FRAME_LIST = 3
        TYPE_STRUCT_LIST = 4

    DF.TYPE_FRAME = AdaptedOrbTypes.TYPE_FRAME
    DF.TYPE_STRUCT = AdaptedOrbTypes.TYPE_STRUCT
    DF.TYPE_FRAME_LIST = AdaptedOrbTypes.TYPE_FRAME_LIST
    DF.TYPE_STRUCT_LIST = AdaptedOrbTypes.TYPE_STRUCT_LIST

    _run_in_gtc = False


def ignore_gtc_check(value=True):
    global _ignore_gtc_check
    _ignore_gtc_check = value


def check_gtc():
    if _ignore_gtc_check:
        return False
    else:
        return _run_in_gtc
