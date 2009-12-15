#
# Copyright 2008-2009 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

# $Id$

__version__ = "$Revision$"
# $Source$

import numina.recipes

def to_json(obj):
    if isinstance(obj, numina.recipes.Parameters):
        return {'__class__': 'numina.recipes.Parameters',
                '__value__': obj.__dict__}
    raise TypeError(repr(obj) + ' is not JSON serializable')

def from_json(obj):
    if '__class__' in obj:
        if obj['__class__'] == 'numina.recipes.Parameters':
            p = numina.recipes.Parameters({}, {}, {}, {}, {})
            p.__dict__ = obj['__value__']
            return p
    return obj