#
# Copyright 2016-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

#cython: language_level=3

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.pycapsule cimport PyCapsule_IsValid
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.pycapsule cimport PyCapsule_GetContext
from cython.operator cimport dereference as deref

cimport cython

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
                      size_t hwin=2, size_t wwin=2, double fill=0.0, reuse_values=False):
    '''Loop over the mask over a window, border is filled with 'fill' value.'''
    cdef:
        size_t xr = arr.shape[1]
        size_t yr = arr.shape[0]
        size_t ii, x, y, xx, yy
        size_t wsize = (2*hwin+1) * (2*wwin+1)
        int status
        size_t bsize = 0
        double *buff[2]
        double* pvalues[3]
        CombineFunc function = NULL
        void *vdata = NULL


    if not PyCapsule_IsValid(method, "numina.cmethod"):
        raise TypeError("parameter is not a valid capsule")

    function = <CombineFunc>PyCapsule_GetPointer(method, "numina.cmethod")
    vdata = PyCapsule_GetContext(method)

    processed = np.zeros_like(badpix, dtype='uint8')
    cdef np.uint8_t[:, ::1] processed_view = processed

    try:
        # Init memory
        for ii in range(2):
            buff[ii] = <double *> PyMem_Malloc(wsize * sizeof(double))
            if not buff[ii]:
                raise MemoryError()

        for ii in range(3):
            pvalues[ii] = <double *> PyMem_Malloc(sizeof(double))
            if not pvalues[ii]:
                raise MemoryError()

        for x in range(xr):
            for y in range(yr):
                bp = badpix[y, x]

                if bp == 0:
                    # Skip over good pixels
                    res[y, x] = <np.double_t>arr[y, x]
                    processed_view[y, x] = 1
                else:
                    # For bad pixels, use a window
                    for xx in range(x-wwin, x+wwin+1):
                        for yy in range(y-hwin, y+hwin+1):
                            if xx < 0 or yy < 0 or xx >= xr or yy >= yr:
                                buff[0][bsize] = fill
                                buff[1][bsize] = 1.0
                                bsize += 1
                            elif badpix[yy, xx] == 0:
                                buff[0][bsize] = arr[yy, xx]
                                buff[1][bsize] = 1.0
                                bsize += 1
                            elif reuse_values and processed_view[yy, xx] == 1:
                                buff[0][bsize] = res[yy, xx]
                                buff[1][bsize] = 1.0
                                bsize += 1
                            else:
                                continue

                    # Compute values
                    if bsize > 0:
                        status = function(buff[0], buff[1], bsize, pvalues, vdata)
                        res[y,x] = <np.double_t>deref(pvalues[0])
                        processed_view[y, x] = 1
                    else:
                        res[y,x] = fill
                    # reset buffers
                    bsize = 0
        return res, processed
    finally:
        # dealocate memory on exit
        for ii in range(2):
            PyMem_Free(buff[ii])
        for ii in range(3):
            PyMem_Free(pvalues[ii])
