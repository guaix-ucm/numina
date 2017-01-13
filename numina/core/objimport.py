#
# Copyright 2011-2014 Universidad Complutense de Madrid
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

"""Import objects by name"""

import importlib
import inspect


def import_object(path):
    """Import an object given its fully qualified name."""
    spl = path.split('.')
    if len(spl) == 1:
        return importlib.import_module(path)
    # avoid last part for the moment
    cls = spl[-1]
    mods = '.'.join(spl[:-1])

    mm = importlib.import_module(mods)
    # try to get the last part as an attribute
    try:
        obj = getattr(mm, cls)
        return obj
    except AttributeError:
        pass

    # Try to import the last part
    rr = importlib.import_module(path)
    return rr


def fully_qualified_name(obj, sep='.'):
    if inspect.isclass(obj):
        return obj.__module__ + sep + obj.__name__
    else:
        return obj.__module__ + sep + obj.__class__.__name__



