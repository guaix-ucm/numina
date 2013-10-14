#
# Copyright 2008-2012 Universidad Complutense de Madrid
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


from __future__ import division

import math

import numpy

from numina.array._nirproc import _process_fowler_intl
from numina.array._nirproc import _process_ramp_intl

def fowler_array(fowlerdata, tint, ts=0, gain=1.0, ron=1.0, 
                badpixels=None, dtype='float64',
                saturation=65631, blank=0, normalize=False):
    '''Loop over the first axis applying Fowler processing.'''
    
    if gain <= 0:
        raise ValueError("invalid parameter, gain <= 0.0")

    if ron <= 0:
        raise ValueError("invalid parameter, ron < 0.0")
    
    if ts <= 0:
        raise ValueError("invalid parameter, ts < 0.0")

    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")

    fowlerdata = numpy.asarray(fowlerdata)
        
    if fowlerdata.ndim != 3:
        raise ValueError('fowlerdata must be 3D')
    
    npairs = fowlerdata.shape[0] // 2
    if 2 * npairs != fowlerdata.shape[0]:
        raise ValueError('axis-0 in fowlerdata must be even')
    
    # change byteorder
    ndtype = fowlerdata.dtype.newbyteorder('=')
    fowlerdata = numpy.asarray(fowlerdata, dtype=ndtype)
    # type of the output
    fdtype = numpy.result_type(fowlerdata.dtype, dtype)
    # Type of the mask
    mdtype = numpy.dtype('uint8')

    fshape = (fowlerdata.shape[1], fowlerdata.shape[2])

    if badpixels is None:
        badpixels = numpy.zeros(fshape, dtype=mdtype)
    else:
        if badpixels.shape != fshape:
            raise ValueError('shape of badpixels is not compatible with shape of fowlerdata')
        if badpixels.dtype != mdtype:
            raise ValueError('dtype of badpixels must be uint8')
            
    result = numpy.empty(fshape, dtype=fdtype)
    var = numpy.empty_like(result)
    npix = numpy.empty(fshape, dtype=mdtype)
    mask = badpixels.copy()

    _process_fowler_intl(fowlerdata, tint, ts,  gain, ron, 
        badpixels, saturation, blank,
        result, var, npix, mask)
    return result, var, npix, mask

def ramp_array(rampdata, tint, tr=0.0, gain=1.0, ron=1.0, 
                badpixels=None, dtype='float64',
                 saturation=65631, nsig=4.0, blank=0, normalize=False):

    if tint <= 0:
        raise ValueError("invalid parameter, tint <= 0.0")

    if gain <= 0:
        raise ValueError("invalid parameter, gain <= 0.0")

    if ron <= 0:
        raise ValueError("invalid parameter, ron < 0.0")

    if nsig <= 0:
        raise ValueError("invalid parameter, nsig <= 0.0")

    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")

    rampdata = numpy.asarray(rampdata)
    if rampdata.ndim != 3:
        raise ValueError('rampdata must be 3D')

    # change byteorder
    ndtype = rampdata.dtype.newbyteorder('=')
    rampdata = numpy.asarray(rampdata, dtype=ndtype)
    # type of the output
    fdtype = numpy.result_type(rampdata.dtype, dtype)
    # Type of the mask
    mdtype = numpy.dtype('uint8')
    fshape = (rampdata.shape[1], rampdata.shape[2])

    if badpixels is None:
        badpixels = numpy.zeros(fshape, dtype=mdtype)
    else:
        if badpixels.shape != fshape:
            raise ValueError('shape of badpixels is not compatible with shape of rampdata')
        if badpixels.dtype != mdtype:
            raise ValueError('dtype of badpixels must be uint8')
            
    result = numpy.empty(fshape, dtype=fdtype)
    var = numpy.empty_like(result)
    npix = numpy.empty(fshape, dtype=mdtype)
    mask = badpixels.copy()

    _process_ramp_intl(rampdata, tint, tr, gain, ron, badpixels, 
        saturation, blank,
        result, var, npix, mask)
    return result, var, npix, mask

def _ramp(data, saturation, dt, gain, ron, nsig):
    nsdata = data[data < saturation]

# Finding glitches in the pixels
    intervals, glitches = _rglitches(nsdata, gain=gain, ron=ron, nsig=nsig)
    vals = numpy.asarray([_slope(nsdata[intls], dt=dt, gain=gain, ron=ron) for intls in intervals if len(nsdata[intls]) >= 2])
    weights = (1.0 / vals[:,1])
    average = numpy.average(vals[:,0], weights=weights)
    variance = 1.0 / weights.sum()
    return average, variance, vals[:,2].sum(), glitches

def _rglitches(nsdata, gain, ron, nsig):
    diffs = nsdata[1:] - nsdata[:-1]
    psmedian = numpy.median(diffs)
    sigma = math.sqrt(abs(psmedian / gain) + 2 * ron * ron)

    start = 0
    intervals = []
    glitches = []
    for idx, diff in enumerate(diffs):
        if not (psmedian - nsig * sigma < diff < psmedian + nsig * sigma):
            intervals.append(slice(start, idx + 1))
            start = idx + 1
            glitches.append(start)
    else:
        intervals.append(slice(start, None))

    return intervals, glitches


def _slope(nsdata, dt, gain, ron):

    if len(nsdata) < 2:
        raise ValueError('Two points needed to compute the slope')

    nn = len(nsdata)
    delt = dt * nn * (nn + 1) * (nn - 1) / 12
    ww = numpy.arange(1, nn + 1) - (nn + 1) / 2
    
    final = (ww * nsdata).sum() / delt
    
    # Readout limited case
    delt2 = dt * delt
    variance1 = (ron / gain)**2 / delt2
    # Photon limiting case
    variance2 = (6 * final * (nn * nn + 1)) / (5 * nn * dt * (nn * nn - 1) * gain)
    variance = variance1 + variance2
    return final, variance, nn
