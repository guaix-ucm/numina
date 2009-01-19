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

class Shutter:
    def __init__(self):
        self.__closed = True 
    
    def status(self,closed = True):
        self.__closed = closed
    
    def open(self):
        if self.__closed:
            self.__closed = False
        else:
            pass
        
    def close(self):
        if not self.__closed:
            self.__closed = True
        else:
            pass
        
    def path(self, input):
        if self.__closed:
            return None
        else:
            return input
        
    def configure(self, open):
        if open:
            self.__closed = False
        else:
            self.__closed = True
    
    def metadata(self):
        return {}
