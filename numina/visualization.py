#
# Copyright 2015-2018 Universidad Complutense de Madrid
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

"""Visualization utilities."""

from __future__ import division

from astropy.visualization import BaseInterval
import numpy as np


class ZScaleInterval(BaseInterval):
    """Compute z1 and z2 cuts in a similar way to Iraf.
    
    If the total number of pixels is less than 10, the function simply returns
    the minimum and the maximum values.

    Parameters
    ----------
    contrast : float, optional
        The scaling factor (between 0 and 1) for determining the minimum
        and maximum value.  Larger values increase the difference
        between the minimum and maximum values used for display.
        Defaults to 0.25.

    .. note:: Deprecated in numina 0.10
          Use `astropy.visualization.ZScaleInterval` instead.
          It will be removed in numina 1.0
    """

    def __init__(self, contrast=0.25):
        self.contrast = contrast

    def get_limits(self, values):
        # Make sure values is a Numpy array
        values = np.asarray(values).ravel()

        npixels = len(values)
        vmin, vmax = np.min(values), np.max(values)

        if npixels < 10:
            return vmin, vmax
        else:
            q375, q500, q625 = np.percentile(values, [37.5, 50.0, 62.5])
            # here 0.25 == 0.625 - 0.375
            # The original algorithm has iterative fitting
            zslope = (q625 - q375) / (0.25 * npixels)
            if self.contrast > 0:
                zslope /= self.contrast
            center = npixels / 2.0
            z1 = q500 - zslope * center
            z2 = q500 + zslope * center

            return max(z1, vmin), min(z2, vmax)

