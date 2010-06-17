#
# Copyright 2008-2010 Sergio Pascual
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

import logging

import numpy

from numina.array.combine import median

_logger = logging.getLogger("numina.array")

def combine_shape(shapes, offsets):
    # Computing final image size and new offsets
    sharr = numpy.asarray(shapes)

    offarr = numpy.asarray(offsets)        
    ucorners = offarr + sharr
    ref = offarr.min(axis=0)     
    finalshape = ucorners.max(axis=0) - ref 
    offsetsp = offarr - ref
    return (finalshape, offsetsp)

def compute_background():
    pass


def resize_array(data, finalshape, region):
    newdata = numpy.zeros(finalshape, dtype=data.dtype)
    newdata[region] = data
    return newdata

def flatcombine(data, masks, scales=None, blank=1.0):
    # Zero masks
    # TODO Do a better fix here
    # This is to avoid negative of zero values in the flat field
    #if scales is not None:
    #    maxscale = max(scales)
    #    scales = [s / maxscale for s in scales]
    
    sf_data = median(data, masks, scales=scales)
    
    mm = sf_data[0] <= 0
    sf_data[0][mm] = blank
    return sf_data

def correct_dark(data, dark, dtype='float32'):
    result = data - dark
    result = result.astype(dtype)
    return result

def correct_flatfield(data, flat, dtype='float32'):
    result = data / flat
    result = result.astype(dtype)
    return result

def correct_nonlinearity(data, polynomial, dtype='float32'):
    result = numpy.polyval(polynomial, data)
    result = result.astype(dtype)
    return result
