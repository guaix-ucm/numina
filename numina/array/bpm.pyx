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

import numpy as np
cimport numpy as np

ctypedef fused image_t:
    double[:,:]
    float[:,:]
    long[:,:]
    int[:,:]

ctypedef fused result_t:
    double[:,:]
    float[:,:]

ctypedef fused mask_t:
    int[:,:]
    char[:,:]

# FIXME: this definition should come from the headers in src
ctypedef int (*CombineFunc)(double*, double*, size_t, double*[3], void*)

@cython.boundscheck(False)
@cython.wraparound(False)
def _process_bpm_intl(object method, image_t arr, mask_t badpix, result_t res,
                      size_t hwin=2, size_t wwin=2, double fill=0.0):
    '''Loop over the mask over a window.'''
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
        int counter

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
                res[y,x] = arr[y, x]
            else:
                # For bad pixels, use a window
                for xx in range(x-wwin, x+wwin+1):
                    for yy in range(y-hwin, y+hwin+1):
                        if xx < 0 or yy < 0 or xx >= xr or yy > yr:
                            buff1[bsize] = fill
                            buff2[bsize] = 1.0
                            bsize += 1
                        elif badpix[yy, xx] == 0:
                            buff1[bsize] = arr[yy, xx]
                            buff2[bsize] = 1.0
                            bsize += 1
                        else:
                            continue
                # Compute something using buff1, buff2
                status = function(buff1, buff2, bsize-1, pvalues, vdata)
                # FIXME: this must have a casting
                res[y,x] = deref(pvalues[0])
                # reset buffer
                bsize = 0
    # dealocate buffer on exit
    free(buff1)
    free(buff2)
    free(pvalues[0])
    free(pvalues[1])
    free(pvalues[2])
    return res
