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

import numpy

from source import LightSource

class Mapper(object):
    def __init__(self, shape):
        self.shape = shape

    def sample(self, source):
        val = source.emit()

        if isinstance(val, (float, numpy.array)):
            return val

        if isinstance(val, LightSource):
            return 0.0

