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

def lookup(name):
    if name == 'json':
        from .jsonserializer import dump, load
        return SerializerInfo('json', dump, load)
    elif name == 'yaml':
        from .yamlserializer import dump, load
        return SerializerInfo('yaml', dump, load)
    else:
        raise LookupError
