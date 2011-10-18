#
# Copyright 2008-2011 Sergio Pascual
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

class Shutter:
    def __init__(self):
        self._closed = True 
    
    def status(self, closed=True):
        self._closed = closed
    
    def open(self):
        if self._closed:
            self._closed = False
        else:
            pass
        
    def close(self):
        if not self._closed:
            self._closed = True
        else:
            pass
        
    def lpath(self, input):
        if self._closed:
            return None
        else:
            return input
        
    def configure(self, open):
        if open:
            self._closed = False
        else:
            self._closed = True
    
    def metadata(self):
        return {}
