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

import simplejson as json

import numina.recipes

def to_json(obj):
    if isinstance(obj, numina.recipes.Parameters):
        return {'__class__': 'numina.recipes.Parameters',
                '__value__': obj.__dict__}
    raise TypeError(repr(obj) + ' is not JSON serializable')

def from_json(obj):
    if '__class__' in obj:
        if obj['__class__'] == 'numina.recipes.Parameters':
            dparam = deunicode_json(obj['__value__'])
            return numina.recipes.Parameters(**dparam)
    return obj

def deunicode_json(obj):
    '''Convert unicode strings into plain strings recursively.'''
    if isinstance(obj, dict):
        newobj = {}
        for key, value in obj.iteritems():
            newobj[str(key)] = deunicode_json(value)
        return newobj
    elif isinstance(obj, list):
        newobj = []
        for i in obj:
            newobj.append(deunicode_json(i))
        return newobj
    elif isinstance(obj, unicode):
        val = str(obj)
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                val = str(val)
        return val
    
    return obj

def param_from_json(name):
    f = open(name)
    try:
        return json.load(f, object_hook=from_json, encoding='utf-8')
    finally:
        f.close()
