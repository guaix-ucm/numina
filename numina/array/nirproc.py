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

import numpy
import math

def axis_fowler(data, badpix, img, var, nmap, mask, hsize, saturation, blank=0):
    '''Apply Fowler processing to a series of data.'''
    MASK_SATURATION = 3 
    MASK_GOOD = 0
     
    if badpix[0] != MASK_GOOD:
        img[...] = blank
        var[...] = blank
        mask[...] = badpix[0]
    else:
        mm = numpy.asarray([(b - a) for a,b in zip(data[:hsize],data[hsize:]) if b < saturation and a < saturation])
        npix = len(mm)
        nmap[...] = npix
        if npix == 0:
            img[...] = blank
            var[...] = blank
            mask[...] = MASK_SATURATION
        elif npix == 1:
            img[...] = mm.mean()
            var[...] = blank
            mask[...] = MASK_GOOD
        else:
            img[...] = mm.mean()
            var[...] = mm.var() / npix
            mask[...] = MASK_GOOD


def axis_ramp(data, badpix, img, var, nmap, mask, crmask, 
              saturation, dt, gain, ron, nsig, blank=0):
    MASK_SATURATION = 3 
    MASK_GOOD = 0

    if badpix[0] != MASK_GOOD:
        img[...] = blank
        var[...] = blank
        mask[...] = badpix[0]
    else:
        mm = data[data < saturation]

        if len(mm) <= 1:
            img[...] = blank
            var[...] = blank
            mask[...] = MASK_SATURATION
        else:
            v, vr, n, glt = ramp(mm, saturation, dt, gain, ron, nsig)

            img[...] = v

            var[...] = vr
            mask[...] = MASK_GOOD
            nmap[...] = n
            # If there is a pixel in the list of CR, put it in the crmask
            if glt:                
                crmask[...] = glt[0]


def fowler_array(data, badpixels, saturation, hsize, blank=0):
    '''Loop over the 3d array applying Fowler processing.'''
    imgfin = None
    varfin = None

#    imgfin = numpy.empty(badpixels.shape, dtype='>i2') # int16, bigendian
#    varfin = numpy.empty(badpixels.shape, dtype='>i2') # int16, bigendian

    nmask = numpy.empty(badpixels.shape, dtype='>u1') # uint8, bigendian
    npixmask = numpy.empty(badpixels.shape, dtype='>u1') # uint8, bigendian

    it = numpy.nditer([data, badpixels, imgfin, varfin, npixmask, nmask], 
                flags=['reduce_ok', 'external_loop',
                    'buffered', 'delay_bufalloc'],
                    op_flags=[['readonly'], ['readonly', 'no_broadcast'], 
                            ['readwrite', 'allocate'], 
                            ['readwrite', 'allocate'],
                            ['readwrite', 'allocate'],
                            ['readwrite', 'allocate'], 
                            ],
                    op_axes=[None,
                            [0,1,-1], 
                            [0,1,-1],
                            [0,1,-1], 
                            [0,1,-1], 
                            [0,1,-1]
                           ])
    for i in range(2, 6):
        it.operands[i][...] = 0
    it.reset()

    for x, badpix, img, var, nmap, mask in it:
        axis_fowler(x, badpix, img, var, nmap, mask, hsize, saturation, blank=blank)

    # Building final frame
    return tuple(it.operands[i] for i in range(2, 6))

def ramp_array(rampdata, dt, gain, ron, badpixels=None, outtype='float64',
                 saturation=60000, nsig=4.0, blank=0):

    outvalue = None
    outvar = None
    npixmask, nmask, ncrs = None, None, None

    if dt <= 0:
        raise ValueError

    if gain <= 0:
        raise ValueError

    if ron <= 0:
        raise ValueError

    if nsig <= 0:
        raise ValueError

    if saturation <= 0:
        raise ValueError

    if badpixels is None:
        badpixels = numpy.zeros((rampdata.shape[0], rampdata.shape[1]), 
                                dtype='uint8')

    fdtype = numpy.result_type(rampdata.dtype, outtype)
    mdtype = 'uint8'

    it = numpy.nditer([rampdata, badpixels, outvalue, outvar, 
                        npixmask, nmask, ncrs], 
                flags=['reduce_ok', 'external_loop',
                    'buffered', 'delay_bufalloc'],
                    op_flags=[['readonly'], ['readonly', 'no_broadcast'], 
                            ['readwrite', 'allocate'], 
                            ['readwrite', 'allocate'],
                            ['readwrite', 'allocate'], 
                            ['readwrite', 'allocate'],
                            ['readwrite', 'allocate'], 
                            ],
                    order='A',
                    op_dtypes=(fdtype, mdtype, fdtype, 
                               fdtype, mdtype, mdtype, mdtype),
                    op_axes=[None,
                            [0,1,-1], 
                            [0,1,-1], 
                            [0,1,-1],
                            [0,1,-1], 
                            [0,1,-1], 
                            [0,1,-1]
                           ])
    for i in range(2, 7):
        it.operands[i][...] = 0
    it.reset()

    for x, badpix, img, var, nmap, mask, crmask in it:
        axis_ramp(x, badpix, img, var, nmap, mask, crmask, 
              saturation, dt, gain, ron, nsig, blank=blank)
        

    # Building final frame

    return tuple(it.operands[i] for i in range(2, 7))

def ramp(data, saturation, dt, gain, ron, nsig):
    nsdata = data[data < saturation]

# Finding glitches in the pixels
    intervals, glitches = rglitches(nsdata, gain=gain, ron=ron, nsig=nsig)
    vals = numpy.asarray([slope(nsdata[intls], dt=dt, gain=gain, ron=ron) for intls in intervals if len(nsdata[intls]) >= 2])
    weights = (1.0 / vals[:,1])
    average = numpy.average(vals[:,0], weights=weights)
    variance = 1.0 / weights.sum()
    return average, variance, vals[:,2].sum(), glitches

def rglitches(nsdata, gain, ron, nsig):
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


def slope(nsdata, dt, gain, ron):

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