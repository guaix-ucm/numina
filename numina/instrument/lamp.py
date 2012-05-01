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

class Lamp(object):
    def __init__(self, radiance):
        self.off = True
        self.radiance = radiance

    def switchon(self):
        self.off = False

    def switchoff(self):
        self.off = True

    def switch(self, off):
        self.off = off

    def emit(self):
        if self.off:
            return 0.0
        else:
            return self.radiance
