#
# Copyright 2008-2011 Universidad Complutense de Madrid
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

class Empty:
    name = 'Empty'
    code = 'Empty'
    def lpath(self, input_):
        return input_

class Filter:
    def __init__(self, name, code, trans, **kwrd):
        self.name = name
        self.code = code
        self.trans = trans
        
    def lpath(self, input_):
        return input_

class FilterWheel:
    def __init__(self, filterlist):
        self._filterlist = filterlist
        self._pos = 0
        self._size = len(filterlist)
        
    def set_position(self, pos):
        self._pos = pos % self._size
        
    def get_current_filter(self):
        return self._filterlist[self._pos]
    
    def lpath(self, input_):
        return self._filterlist[self._pos].path(input_)
    
    def get_size(self):
        return self._size
    
    def metadata(self):
        return {'FILTER': self._filterlist[self._pos].code}
