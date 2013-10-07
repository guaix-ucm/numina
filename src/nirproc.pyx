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

def process_fowler_17(arr):
    ndtype = arr.dtype.newbyteorder('=')
    arr = np.asarray(arr, dtype=ndtype)
    fshape = (arr.shape[1], arr.shape[2])
    ftype = 'float32'
    
    res = np.empty(fshape, dtype=ftype)
    
    process_fowler_16(arr, res)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def process_fowler_16(datacube_t arr, result_t res, mask_t badpix=None, 
                double saturation=55000.0):
    cdef:
        size_t xr = arr.shape[2]
        size_t yr = arr.shape[1]
        size_t zr = arr.shape[0]
        size_t x, y, z
        size_t ll = 0
        size_t h = zr // 2
        result_t var
        mask_t npix, mask
        double rsl, vrr
        char npx, msk, bp

    cdef vector[double] vect
    vect.reserve(zr)

    if badpix is None:
        badpix = np.zeros((yr, xr), dtype='uint8')

    res = np.empty((yr, xr))
    var = np.empty_like(res)
    npix = np.empty((yr, xr), dtype='uint8')
    mask = badpix.copy()
    # We are not checking than arr and res shapes are compatible!
    
    
    for x in range(xr):
        for y in range(yr):
            bp = badpix[y, x]
            if bp == 0:
                for z in range(h):
                    rsl = arr[z,y,x]
                    vrr = arr[z+h,y,x]
                    if rsl < saturation and vrr < saturation:
                        vect.push_back(vrr-rsl)

                axis_fowler_6(vect, rsl, vrr, npx, msk)
            else:
                rsl = vrr = 0.0
                npx = 0
                msk = bp

            res[y,x] = rsl
            var[y,x] = vrr
            npix[y,x] = npx
            mask[y,x] = msk
            vect.clean()

    return res, var, npix, mask

# This is a pure C++ function, it could be defined elsewhere
# and imported in cython
cdef double axis_fowler_6(vector[double] buff, double& res, double &var, char& npx, char& mask):
    (&npx)[0] = buff.size()
    if npx == 0:
        #all is saturated
        # Ugly workaround
        (&res)[0] = 1.0
        (&var)[0] = 2.0
        (&mask)[0] = 3
    else:
        # Ugly workaround
        (&res)[0] = 4.0
        (&var)[0] = 1.0
        (&mask)[0] = 0
        
    return 0.0

def fowler_array(fowlerdata, badpixels=None, dtype='float64',
                 saturation=65631, blank=0):
    '''Loop over the 3d array applying Fowler processing.'''
    
    
    fowlerdata = np.asarray(fowlerdata)
        
    if fowlerdata.ndim != 3:
        raise ValueError('fowlerdata must be 3D')
    
    hsize = fowlerdata.shape[2] // 2
    if 2 * hsize != fowlerdata.shape[2]:
        raise ValueError('axis-2 in fowlerdata must be even')
    
    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")
    
    if badpixels is None:
        badpixels = np.zeros((fowlerdata.shape[0], fowlerdata.shape[1]), 
                                dtype='uint8')

    fdtype = np.result_type(fowlerdata.dtype, dtype)
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
        badpixels = np.zeros((rampdata.shape[0], rampdata.shape[1]), 
                                dtype='uint8')

    fdtype = np.result_type(rampdata.dtype, dtype)
    mdtype = 'uint8'

    outvalue = None
    outvar = None
    npixmask, nmask, ncrs = None, None, None


