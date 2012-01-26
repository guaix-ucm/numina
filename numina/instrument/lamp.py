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

# Classes are new style
__metaclass__ = type

class Lamp:
    def __init__(self, name, code, source):
        self.name = name
        self.code = code
        self._source = source
        self.power = False
    
    @property
    def source(self):
        if self.power:
            return self._source
        else:
            return None
    

class Lamps:
    def __init__(self, lamplist):
        self.lamplist = lamplist
    
    @property
    def source(self):
        for i in self.lamplist:
            if i.power:
                return i.source
    
    def switch(self, status):
        for i in self.lamplist:
            i.power = status
