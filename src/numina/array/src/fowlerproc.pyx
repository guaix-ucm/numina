#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

#cython: language_level=3

cimport cython
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

cdef extern from "nu_fowler.h" namespace "Numina":
    cdef cppclass FowlerResult[T]:
        FowlerResult() except +
        T value
        T variance
        char npix
        char mask

# Generic, valid with teff > 0
    FowlerResult[double] axis_fowler(vector[double] buff, double teff, double gain, double ron, double ts, double blank)
# RON limited case
    FowlerResult[double] axis_fowler_ron(vector[double] buff, double teff, double gain, double ron, double ts, double blank)
    ctypedef FowlerResult[double] (*axis_fowler_func_t)(vector[double] buff, double teff, double gain, double ron, double ts, double blank)

@cython.boundscheck(False)
@cython.wraparound(False)
def _process_fowler_intl(datacube_t arr, double tint, double ts, double gain, double ron, mask_t badpix, double saturation, double blank,
        result_t res, 
        result_t var, 
        mask_t npix,
        mask_t mask
        ):
    '''Loop over the first axis applying Fowler processing.'''
    cdef:
        size_t xr = arr.shape[2]
        size_t yr = arr.shape[1]
        size_t zr = arr.shape[0]
        size_t x, y, z
        double teff
        size_t np = zr // 2
        FowlerResult[double] fres
        double val1, val2
        char bp 
        vector[double] buff
        axis_fowler_func_t axis_func = axis_fowler

    # integration time minus sample times number of pairs
    teff = tint - (np - 1) * ts
    if teff <= 0:
        # where are in a RON limited case
        axis_func = axis_fowler_ron

    buff.reserve(zr)

    for x in range(xr):
        for y in range(yr):
            bp = badpix[y, x]
            if bp == MASK_GOOD:
                for z in range(np):
                    val1 = arr[z,y,x]
                    val2 = arr[z+np,y,x]
                    if val1 < saturation and val2 < saturation:
                        buff.push_back(val2 - val1)

                fres = axis_func(buff, teff, gain, ron, ts, blank)
            else:
                fres.value = fres.variance = blank
                fres.npix = 0
                fres.mask = bp

            res[y,x] = fres.value
            var[y,x] = fres.variance
            npix[y,x] = fres.npix
            mask[y,x] = fres.mask
            buff.clear()

    return res, var, npix, mask

