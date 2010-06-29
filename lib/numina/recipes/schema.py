
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

import collections

Schema = collections.namedtuple('Schema', 'name value description')

_data = {
            'extinction': Schema('extinction', 0.0, 'Atmospheric extinction'),
            'nonlinearity': Schema('nonlinearity', [1.0, 0.0], 'Non-linearity correction'),
            'master_dark': Schema('master_dark', None, 'Master dark'),
            'master_bpm': Schema('master_bpm', None, 'Master bad pixel mask'),
            'master_bias': Schema('master_bias', None, 'Master bias'),
            'master_flat': Schema('master_flat', None, 'Master flat'),
            'nthreads': Schema('nthreads', 1, 'Nunber of threads'),
        }

class BaseSchema(object):
    def __init__(self, data):
        self._data = data

    def keys(self):
        return self._data.keys()

    def lookup(self, parameter):
        return self._data.get(parameter)

_schema = [BaseSchema(_data)]

_undefined = Schema('<name>', None, None)

def undefined(name):
    return _undefined._replace(name=name)

def lookup(parameter):
    for s in _schema:
        defc = s.lookup(parameter)
        if defc is not None:
            break
    return defc

def list():
    result = {}
    for r in _schema:
        keys = r.keys()
        for k in keys:
            schema = lookup(k)
            if schema is None:
                # Schema not defined
                schema = undefined(k)
            result[k] = schema
    return [val for val in result.itervalues()]
