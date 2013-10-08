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

import numpy as np
cimport numpy as cnp

cimport cython
from libcpp.vector cimport vector

ctypedef fused datacube_t:
    double[:,:,:]
    float[:,:,:]
    long[:,:,:]
    int[:,:,:]

ctypedef fused result_t:
    double[:,:]
    float[:,:]

ctypedef char[:,:] mask_t

DEF NFRAME = 100

cdef extern from "nu_fowler.h" namespace "NuminaAlt":
    cdef cppclass FowlerResult[T]:
        FowlerResult() except +
        T value
        T variance
        char npix
        char mask

    FowlerResult[double] axis_fowler(vector[double] buff)

def fowler_array(fowlerdata, badpixels=None, dtype='float64',
                 saturation=65631, blank=0):
    '''Loop over the 3d array applying Fowler processing.'''
    
    fowlerdata = np.asarray(fowlerdata)
        
    if fowlerdata.ndim != 3:
        raise ValueError('fowlerdata must be 3D')
    
    hsize = fowlerdata.shape[0] // 2
    if 2 * hsize != fowlerdata.shape[0]:
        raise ValueError('axis-0 in fowlerdata must be even')
    
    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")
    
    # change byteorder
    ndtype = fowlerdata.dtype.newbyteorder('=')
    fowlerdata = np.asarray(fowlerdata, dtype=ndtype)
    fdtype = np.result_type(fowlerdata.dtype, dtype)
    mdtype = 'uint8'

    fshape = (fowlerdata.shape[1], fowlerdata.shape[2])

    if badpixels is None:
        badpixels = np.zeros(fshape, dtype=mdtype)

    result = np.empty(fshape, dtype=fdtype)
    var = np.empty_like(result)
    npix = np.empty(fshape, dtype=mdtype)
    mask = badpixels.copy()

    process_fowler_intl(fowlerdata, badpixels, saturation, 
        result, var, npix, mask)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def process_fowler_intl(datacube_t arr, mask_t badpix, double saturation, 
        result_t res, 
        result_t var, 
        mask_t npix,
        mask_t mask
        ):
    cdef:
        size_t xr = arr.shape[2]
        size_t yr = arr.shape[1]
        size_t zr = arr.shape[0]
        size_t x, y, z
        size_t h = zr // 2
        FowlerResult[double] fres
        double rsl, vrr
        char bp

    cdef vector[double] vect
    vect.reserve(zr)

    for x in range(xr):
        for y in range(yr):
            bp = badpix[y, x]
            if bp == 0:
                for z in range(h):
                    rsl = arr[z,y,x]
                    vrr = arr[z+h,y,x]
                    if rsl < saturation and vrr < saturation:
                        vect.push_back(vrr-rsl)

                fres = axis_fowler(vect)
            else:
                fres.value = fres.variance = 0.0
                fres.npix = 0
                fres.mask = bp

            res[y,x] = fres.value
            var[y,x] = fres.variance
            npix[y,x] = fres.npix
            mask[y,x] = fres.mask
            vect.clean()

    return res, var, npix, mask


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
        badpixels = np.zeros((rampdata.shape[0], rampdata.shape[1]), 
                                dtype='uint8')

    fdtype = np.result_type(rampdata.dtype, dtype)
    mdtype = 'uint8'

    outvalue = None
    outvar = None
    npixmask, nmask, ncrs = None, None, None


