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

class Empty:
    name = 'Empty'
    code = 'Empty'
    def path(input):
        return input

class Filter:
    def __init__(self, name, code, trans, **kwrd):
        self.name = name
        self.code = code
        self.trans = trans
        
    def path(self,input):
        return input

class FilterWheel:
    def __init__(self, filterlist):
        self.__filterlist = filterlist
        self.__pos = 0
        self.__size = len(filterlist)
        
    def set_position(self,pos):
        self.__pos = pos % self.__size
        
    def get_current_filter(self):
        return self.__filterlist[self.__pos]
    
    def path(self,input):
        return self.__filterlist[self.__pos].path(input)
    
    def get_size(self):
        return self.__size
    
    def metadata(self):
        return {'FILTER': self.__filterlist[self.__pos].code}
