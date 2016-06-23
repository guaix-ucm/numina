#
# Copyright 2008-2014 Universidad Complutense de Madrid
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
    # FIXME: workaround
    DF = types.ModuleType('DF')
    DF.TYPE_FRAME = None
    _run_in_gtc = False

_eqtypes = {'numina.core.products.FrameDataProduct': DF.TYPE_FRAME}


def dialect_info(obj):
    key = obj.__module__ + '.' + obj.__class__.__name__
    tipo = _eqtypes.get(key, None)
    result = {'gtc': {'fqn': key, 'python': obj.python_type, 'type': tipo}}
    return result


def register(more):
    _eqtypes.update(more)


def ignore_gtc_check(value=True):
    global _ignore_gtc_check
    _ignore_gtc_check = value


def check_gtc():
    if _ignore_gtc_check:
        return False
    else:
        return _run_in_gtc
