#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

'''Serialize objects using JSON.'''

import json

def to_json(obj):
    if hasattr(obj, '__getstate__'):
        dparam = obj.__getstate__()
    else:
        try:
            dparam = obj.__dict__
        except AttributeError:
            raise TypeError('%s is not JSON serializable' % obj.__class__)
    return {'__class__': obj.__class__.__name__,
            '__module__': obj.__class__.__module__,
            '__value__': dparam,
            }

def obj_is_serialized_class(obj):
    '''Check if obj represents a serialized class.'''
    if '__class__' in obj and '__module__' in obj and '__value__' in obj:
        return True
    return False


def from_json(obj):
    if obj_is_serialized_class(obj):
        clsname = obj['__class__']
        modname = obj['__module__']
        _mod = __import__(modname, globals(), locals(), [clsname], -1)
        cls = getattr(_mod, clsname)
        result = super(type(cls), cls).__new__(cls)
        
        dparam = deunicode_json(obj['__value__'])
        
        if hasattr(result, '__setstate__'):
            result.__setstate__(dparam)
        else:
            result.__dict__ = dparam
        return result
    elif isinstance(obj, dict):
        return deunicode_json(obj)
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


def dump(data, fd):
    json.dump(data, fd, indent=1, default=to_json)


load = json.load