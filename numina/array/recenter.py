#
# Copyright 2013-2015 Universidad Complutense de Madrid
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

import numpy
from scipy.spatial import distance

from numina.array.utils import image_box


# returns y,x
def _centering_centroid_loop(data, center, box):
    # extract raster image
    sl = image_box(center, data.shape, box)
    raster = data[sl]

    # Background estimation for recentering
    background = raster.min()

    braster = raster - background
    threshold = braster.mean()
    mask = braster >= threshold
    if not numpy.any(mask):
        return center, background

    rr = numpy.where(mask, braster, 0)
    norm = rr.sum()
    # All points in thresholded raster are 0.0
    if norm <= 0.0:
        return center, background

    fi, ci = numpy.indices(braster.shape)

    fm = (rr * fi).sum() / norm
    cm = (rr * ci).sum() / norm

    return (fm + sl[0].start, cm + sl[1].start), background


def _centering_centroid_loop_xy(data, center_xy, box):
    center_yx = center_xy[::-1]

    ncenter_yx, back = _centering_centroid_loop(data, center_yx, box)
    ncenter_xy = ncenter_yx[::-1]
    return ncenter_xy, back


def centering_centroid(data, xi, yi, box, nloop=10, toldist=1e-3,
                       maxdist=10.0):
    '''
        returns x, y, background, status, message

        status is:
          * 0: not recentering
          * 1: recentering successful
          * 2: maximum distance reached
          * 3: not converged
    '''

    # Store original center
    cxy = (xi, yi)
    origin = (xi, yi)
    # initial background
    back = 0.0

    if nloop == 0:
        return xi, yi, 0.0, 0, 'not recentering'

    for i in range(nloop):
        nxy, back = _centering_centroid_loop_xy(data, cxy, box)
        # _logger.debug('new center is %s', ncenter)
        # if we are to far away from the initial point, break
        dst = distance.euclidean(origin, nxy)
        if dst > maxdist:
            msg = 'maximum distance (%5.2f) from origin reached' % maxdist
            return cxy[0], cxy[1], back, 2, msg

        # check convergence
        dst = distance.euclidean(nxy, cxy)
        if dst < toldist:
            return nxy[0], nxy[1], back, 1, 'converged in iteration %i' % i
        else:
            cxy = nxy

    return nxy[0], nxy[1], back, 3, 'not converged in %i iterations' % nloop
