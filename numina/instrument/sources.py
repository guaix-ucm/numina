#
# Copyright 2008-2013 Universidad Complutense de Madrid
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


class MultiSource(object):
    def __init__(self, *sources):
        self.sources = sources

    def emit(self):
        return sum(src.emit() for src in self.sources)

class LightSource(object):
    
    def __add__(self, other):
        return MultiSource(self, other)

class Sky(object):
    def __init__(self, radiance):
        self.radiance = radiance
        
    def emit(self):
        return self.radiance

class ThermalBackground(object):
    def __init__(self, radiance):
        self.radiance = radiance
        
    def emit(self):
        return self.radiance