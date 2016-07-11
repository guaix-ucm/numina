#
# Copyright 2016 Universidad Complutense de Madrid
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

from libc.stdlib cimport malloc, free

from cpython.pycapsule cimport PyCapsule_IsValid
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.pycapsule cimport PyCapsule_GetContext
from cython.operator cimport dereference as deref
cimport cython
from cpython cimport bool

import numpy as np
cimport numpy as np


ctypedef fused image_t:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t
    np.int16_t
    np.int8_t
    np.uint64_t
    np.uint32_t
    np.uint16_t
    np.uint8_t

# FIXME: this definition should come from the headers in src
ctypedef int (*CombineFunc)(double*, double*, size_t, double*[3], void*)

@cython.boundscheck(False)
@cython.wraparound(False)
def _process_bpm_intl(object method, image_t[:,:] arr, np.uint8_t[:,:] badpix, np.double_t[:,:] res,
                      size_t hwin=2, size_t wwin=2, double fill=0.0):
    '''Loop over the mask over a window, border is filled with 'fill' value.'''
    cdef:
        size_t xr = arr.shape[1]
        size_t yr = arr.shape[0]
        size_t x, y, xx, yy
        int status
        char bp
        size_t bsize = 0
        double *buff1
        double *buff2
        double* pvalues[3]
        CombineFunc function
        void *vdata

    buff1 = <double *>malloc((2*hwin+1) * (2*wwin+1) * sizeof(double))
    buff2 = <double *>malloc((2*hwin+1) * (2*wwin+1) * sizeof(double))
    pvalues[0] = <double *>malloc(sizeof(double))
    pvalues[1] = <double *>malloc(sizeof(double))
    pvalues[2] = <double *>malloc(sizeof(double))

    function = NULL
    vdata = NULL

    if not PyCapsule_IsValid(method, "numina.cmethod"):
        raise TypeError("parameter is not a valid capsule")

    function = <CombineFunc>PyCapsule_GetPointer(method, "numina.cmethod")
    vdata = PyCapsule_GetContext(method)

    for x in range(xr):
        for y in range(yr):
            bp = badpix[y, x]

            if bp == 0:
                # Skip over good pixels
                res[y,x] = <np.double_t>arr[y, x]
            else:
                # For bad pixels, use a window
                for xx in range(x-wwin, x+wwin+1):
                    for yy in range(y-hwin, y+hwin+1):
                        if xx < 0 or yy < 0 or xx >= xr or yy >= yr:
                            buff1[bsize] = fill
                            buff2[bsize] = 1.0
                            bsize += 1
                        elif badpix[yy, xx] == 0:
                            buff1[bsize] = arr[yy, xx]
                            buff2[bsize] = 1.0
                            bsize += 1
                        else:
                            continue
                # Compute value using buff1, buff2

                status = function(buff1, buff2, bsize, pvalues, vdata)

                res[y,x] = <np.double_t>deref(pvalues[0])
                # reset buffer
                bsize = 0

    # dealocate buffer on exit
    free(buff1)
    free(buff2)
    free(pvalues[0])
    free(pvalues[1])
    free(pvalues[2])
    return res
