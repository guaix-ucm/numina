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


import pkgutil

try:
    import DF
except ImportError:
    import types
    # FIXME: workaround
    DF = types.ModuleType('DF')
    DF.TYPE_FRAME = None

_eqtypes = {'numina.core.products.FrameDataProduct': DF.TYPE_FRAME}


def dialect_info(obj):
    key = obj.__module__ + '.' + obj.__class__.__name__
    tipo = _eqtypes.get(key, None)
    result = {'gtc': {'fqn': key, 'python': obj.python_type, 'type': tipo}}
    return result


def register(more):
    _eqtypes.update(more)
