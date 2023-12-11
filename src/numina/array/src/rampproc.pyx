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

cdef extern from "nu_ramp.h" namespace "Numina":
    cdef cppclass RampResult[T]:
        RampResult() except +
        T value
        T variance
        char npix
        char mask

    cdef cppclass HWeights:
        pass

    ctypedef vector[HWeights] HWeightsStore
    HWeightsStore create(size_t n)
    RampResult[double] axis_ramp(vector[double] buff, double dt, 
        double gain, double ron, HWeightsStore wgts_store, double blank)

@cython.boundscheck(False)
@cython.wraparound(False)
def _process_ramp_intl(datacube_t arr, double tint, double gain, double ron, mask_t badpix, double saturation, double blank,
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
        RampResult[double] fres
        double val
        double dt
        char bp 
        vector[double] buff
        # weights and internal values
        vector[HWeights] wgts

    dt = tint / (zr - 1)

    buff.reserve(zr)
    wgts = create(zr)

    for x in range(xr):
        for y in range(yr):
            bp = badpix[y, x]
            if bp == MASK_GOOD:
                for z in range(zr):
                    val = arr[z,y,x]
                    if val < saturation:
                        buff.push_back(val)
                    else:
                        # We stop collecting at saturation level
                        break

                fres = axis_ramp(buff, dt, gain, ron, wgts, blank)
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

