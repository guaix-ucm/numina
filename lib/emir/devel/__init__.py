#
# Copyright 2008 Sergio Pascual
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

import numpy

def somegauss(shape, level, gaussians = None):
  '''Some gaussians in an image'''
  intensity = 10.0
  im = level * numpy.ones(shape)
  if gaussians is not None:
      y, x = numpy.indices(shape)
      for i in gaussians:
          x0 = i[0]
          y0 = i[1]
          sigma0 = i[2]
          intensity = i[3]
          im +=  intensity * numpy.exp(-((x - x0)**2+(y - y0)**2)/(sigma0)**2)
  
  return im
