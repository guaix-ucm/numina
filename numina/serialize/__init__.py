#
# Copyright 2012 Universidad Complutense de Madrid
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

'''Base classes for serialization'''

from collections import namedtuple
    
SerializerInfo = namedtuple('SerializerInfo', ['name', 'dump', 'load'])

_serializers = {}

def register(name, dump, load):
    '''Register a serializer by name.'''
    global _serializers
    _serializers[name] = (dump, load)
    
def unregister(name):
    '''Unregister a serializer by name.'''
    global _serializers
    if name in _serializers:
        del _serializers[name]


def lookup(name):
    '''Lookup a serializer by name.'''
    if name == 'json':
        from .jsonserializer import dump as jdump
        from .jsonserializer import load as jload
        return SerializerInfo('json', jdump, jload)
    elif name == 'yaml':
        from .yamlserializer import dump as ydump
        from .yamlserializer import load as yload
        return SerializerInfo('yaml', ydump, yload)
    elif name in _serializers:
        dump, load = _serializers[name]
        return SerializerInfo(name, dump, load)
    else:
        raise LookupError
