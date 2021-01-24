#
# Copyright 2008-2021 Universidad Complutense de Madrid
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

