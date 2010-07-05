
#
# Copyright 2010 Sergio Pascual
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

from __future__ import with_statement

import simplejson as json

from numina.jsonserializer import from_json
import schema

class DictRepo(object):
    def __init__(self, dicto):
        self._data = dicto

    def lookup(self, uid, parameter):
        return self._data.get(parameter)

    def keys(self):
        return self._data.keys()

class JSON_Repo(object):
    def __init__(self, filename):
        with open(filename, mode='r') as f:
            self._data = json.load(f, object_hook=from_json, encoding='utf-8')

    def lookup(self, uid, parameter):
        return self._data.get(parameter)

    def keys(self):
        return self._data.keys()

_repos = [DictRepo({'hare': 90, 'linearity':[1.0, 0.0], 'master_dark':'dum.fits'})]

def get_repo_list():
    return _repos

def set_repo_list(newlist):
    global _repos
    result = _repos
    _repos = newlist
    assert id(newlist) != id(result)
    return result

def list_keys():
    result = {}
    for r in _repos:
        keys = r.keys()
        for k in keys:
            sch = schema.lookup(k)
            if sch is None:
                # Schema not defined
                sch = schema.undefined(k)
            result[k] = sch
    return [val for val in result.itervalues()]
        

def lookup(uid, parameter):
    defc = schema.lookup(parameter)

    for r in _repos:
        val = r.lookup(uid, parameter)
        if val is not None:
            return val

    if defc is not None:
        return defc.value

    raise LookupError('Parameter %s not found in registry' % parameter)
