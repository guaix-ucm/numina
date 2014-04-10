#
# Copyright 2013-2014 Universidad Complutense de Madrid
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

'''Recenter routines'''

from __future__ import division

import logging
import math

import numpy
from scipy.spatial import distance


def img_box(center, shape, box):

    def slice_create(c, s, b):
        cc = int(math.floor(c + 0.5))
        l = max(0, cc - b)
        h = min(s, cc + b +1)
        return slice(l, h, None)

    return tuple(slice_create(*args) for args in zip(center, shape, box))

# returns y,x
def _centering_centroid_loop(data, center, box):
    sl = img_box(center, data.shape, box)
    
    raster = data[sl]
    
    background = raster.min()
    
    braster = raster - background

    threshold = braster.mean()
    
    mask = braster >= threshold
    if not numpy.any(mask):
        return center
        
    rr = numpy.where(mask, braster, 0)

    #r_std = rr.std()
    #r_mean = rr.mean()
    #if r_std > 0:
    #    snr = r_mean / r_std
    
    fi, ci = numpy.indices(braster.shape)
    
    norm = rr.sum()
    if norm <= 0.0:
        #_logger.warning('all points in thresholded raster are 0.0')
        return center
        
    fm = (rr * fi).sum() / norm
    cm = (rr * ci).sum() / norm
    
    return fm + sl[0].start, cm + sl[1].start
    
# returns y,x
def centering_centroid(data, center, box, nloop=10, toldist=1e-3, maxdist=10):
    
    # Store original center
    ocenter = center.copy()
    
    for i in range(nloop):
        
        ncenter = _centering_centroid_loop(data, center, box)
        # if we are to far away from the initial point, break
        dst = distance.euclidean(ocenter, ncenter)
        if dst > maxdist:
            msg = 'maximum distance (%i) from origin reached' % maxdist 
            return center, msg
        
        # check convergence
        dst = distance.euclidean(ncenter, center)
        if dst < toldist:
            msg = 'converged in iteration %i' % i
            return ncenter, msg
        else:
            center = ncenter
        
    msg = 'not converged in %i iterations' % nloop
    return ncenter, msg

