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

class Lamp:
    def __init__(self, name, code, source):
        self.name = name
        self.code = code
        self.__source = source
        self.power = False
        
    def __getS(self):
        if self.power:
            return self.__source
        else:
            return None
    
    source = property(__getS)
    

class Lamps:
    def __init__(self, lamplist):
        self.lamplist = lamplist
    
    @property
    def source(self):
        for i in self.lamplist:
            if i.power:
                return i.source
    
    def switch(status):
        for i in self.lamplist:
            i.power = status
