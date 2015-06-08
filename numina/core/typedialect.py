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
import numina.ext as namespace


def default_dialect_info(obj):
    key = obj.__module__ + '.' + obj.__class__.__name__
    result = {'base': {'fqn': key, 'python': obj.python_type}}
    return result


for imp, name, _is_pkg in pkgutil.walk_packages(namespace.__path__,
                                                namespace.__name__ + '.'):
    try:
        loader = imp.find_module(name)
        mod = loader.load_module(name)
        dialect_info = getattr(mod, 'dialect_info')
        if dialect_info:
            break
    except Exception as error:
        # print name, type(error), error
        pass
else:
    dialect_info = default_dialect_info
