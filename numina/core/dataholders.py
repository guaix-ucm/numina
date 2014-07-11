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

'''
Recipe requirements
'''

import inspect

from .types import NullType, PlainPythonType
from .types import ListOfType


class EntryHolder(object):
    def __init__(self, tipo, description, destination, optional, default):
        if tipo is None:
            self.type = NullType()
        elif tipo in [bool, str, int, float, complex]:
            self.type = PlainPythonType(ref=tipo())
        elif isinstance(tipo, ListOfType):
            self.type = tipo
        elif inspect.isclass(tipo):
            self.type = tipo()
        else:
            self.type = tipo
            
        self.description = description
        self.optional = optional
        self.dest = destination
        self.default = default


