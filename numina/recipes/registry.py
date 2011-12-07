
#
# Copyright 2010-2011 Universidad Complutense de Madrid
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

import collections
import json

from numina.jsonserializer import from_json

Schema = collections.namedtuple('Schema', 'name value description')

_rr = dict(recipe={'parameters' : {},
                    'run': {'repeat':1,
                            'threads': 1,
                            'instrument': None,
                            'mode': None}
                    }, 
            observing_block={'instrument': None,
                             'mode': None,
                             'id': -1,
                             'result': {}},
            )

class ProxyPath(object):
    def __init__(self, path):
        self.path = path
        self._split = filter(None, path.split('/'))

    def get(self):
        global _rr
        return _re_get(_rr, self._split)

class ProxyQuery(object):
    def __init__(self, dummy=None):
        self.dummy = dummy
        
    def get(self):
        return self.dummy
    
def init_registry():    
    pass

def init_registry_from_file(filename):
    global _rr
    with open(filename, mode='r') as ff:
        rr = json.load(ff, object_hook=from_json, encoding='utf-8')
    
    for keys, val in _re_list(rr, ''):
        _re_set(_rr, keys[:-1], keys[-1], val)

def print_registry():
    _re_print(_rr, '')

def _re_print(obj, prefix):
    for key in obj:
        if isinstance(obj[key], dict):
            newpre = prefix + '/' + key
            _re_print(obj[key], newpre)
        else:
            print prefix + '/' + key, ':', obj[key]

def list_registry():
    global _rr
    return _re_list(_rr, [])

def _re_list(obj, prefix):
    if obj and isinstance(obj, dict):
        result = []
        for key in obj:
            newpre = list(prefix)
            newpre.append(key)
            result.extend(_re_list(obj[key], newpre))
        return result
    else:
        return [(prefix, obj)]

def _re_set(obj, path, key, value):
    if not path:
        if not isinstance(value, dict) and obj and obj.has_key(key) and isinstance(obj[key], dict):
            raise TypeError('this path should be a directory')
        if not isinstance(obj, dict):
            raise TypeError('this path should be a directory')
        obj[key] = value
        return
    else:
        if isinstance(obj, dict):
            newkey, newpath = path[0], path[1:]
            if not obj.has_key(newkey):
                obj[newkey] = {}
            _re_set(obj[newkey], newpath, key, value)
        else:
            raise TypeError('path too short')

def set(path, value):
    global _rr
    split = filter(None, path.split('/'))
    if not split:
        raise TypeError('can not set value on /')
    newpath, newkey = split[:-1], split[-1]
    _re_set(_rr, newpath, newkey, value)


def get(path):
    global _rr
    split = filter(None, path.split('/'))
    try:
        return _re_get(_rr, split)
    except KeyError:
        raise KeyError(path)
    
def mget(paths):
    from itertools import islice, ifilter, imap
    
    def helper(path):
        try:
            return get(path)
        except KeyError:
            return None
        
    result = list(islice(ifilter(None, imap(helper, paths)), 1))
    if not result:
        raise KeyError('paths %s' % paths)
    return result[0]
    

def _re_get(obj, path):
    if path and isinstance(obj, dict):
        return _re_get(obj[path[0]], path[1:])
    return obj

if __name__ == '__main__':
    
    init_registry_from_file('/home/spr/Datos/emir/apr21/config.txt')

    print get('/recipe')
    

