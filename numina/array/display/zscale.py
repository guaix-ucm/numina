#
# Copyright 2015 Universidad Complutense de Madrid
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
from __future__ import print_function

import numpy as np


def zscale(image, factor=0.25, debug=False):
    """Compute z1 and z2 cuts in a similar way to Iraf.
    
    If the total number of pixels is less than 10, the function simply 
    returns the minimum and the maximum values.

    Parameters
    ----------
    image : np.ndarray
        Image array.
    factor : float
        Factor.
    debug : bool
        If True displays computed values.

    Returns
    -------
    z1 : float
        Background value.
    z2 : float
        Foreground value.

    """

    # protections
    if type(image) is not np.ndarray:
        raise ValueError('image=' + str(image) + ' must be a numpy.ndarray')

    npixels = image.size

    if npixels < 10:
        z1 = np.min(image)
        z2 = np.max(image)
    else:
        fnpixels = float(npixels)
        perclist = [00.0, 37.5, 50.0, 62.5, 100.0]
        q000, q375, q500, q625, q1000 = np.percentile(image, perclist)
        zslope = (q625-q375)/(0.25*fnpixels)
        z1 = q500-(zslope*fnpixels/2)/factor
        z1 = max(z1, q000)
        z2 = q500+(zslope*fnpixels/2)/factor
        z2 = min(z2, q1000)
    
    if debug:
        print('>>> z1, z2:', z1, z2)

    return z1, z2
