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

import numpy

def fowler_array(fowlerdata, badpixels=None, dtype='float64',
                 saturation=65631, blank=0):
    '''Loop over the 3d array applying Fowler processing.'''
    
    
    fowlerdata = numpy.asarray(fowlerdata)
        
    if fowlerdata.ndim != 3:
        raise ValueError('fowlerdata must be 3D')
    
    hsize = fowlerdata.shape[2] // 2
    if 2 * hsize != fowlerdata.shape[2]:
        raise ValueError('axis-2 in fowlerdata must be even')
    
    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")
    
    if badpixels is None:
        badpixels = numpy.zeros((fowlerdata.shape[0], fowlerdata.shape[1]), 
                                dtype='uint8')

    fdtype = numpy.result_type(fowlerdata.dtype, dtype)
    mdtype = 'uint8'

    outvalue = None
    outvar = None
    npixmask, nmask = None, None

def ramp_array(rampdata, dt, gain, ron, badpixels=None, dtype='float64',
                 saturation=65631, nsig=4.0, blank=0):

    if dt <= 0:
        raise ValueError("invalid parameter, dt <= 0.0")

    if gain <= 0:
        raise ValueError("invalid parameter, gain <= 0.0")

    if ron <= 0:
        raise ValueError("invalid parameter, ron < 0.0")

    if nsig <= 0:
        raise ValueError("invalid parameter, nsig <= 0.0")

    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")

    if badpixels is None:
        badpixels = numpy.zeros((rampdata.shape[0], rampdata.shape[1]), 
                                dtype='uint8')

    fdtype = numpy.result_type(rampdata.dtype, dtype)
    mdtype = 'uint8'

    outvalue = None
    outvar = None
    npixmask, nmask, ncrs = None, None, None


