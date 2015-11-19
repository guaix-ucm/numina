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


# ------------------------------------------------------------------------------

def sigmaG(x):
    """Compute a robust estimator of the standard deviation

    See Eq. 3.36 (page 84) in Statistics, Data Mining, and Machine
    in Astronomy, by Ivezic, Connolly, VanderPlas & Gray

    Parameters
    ----------
    x : 1d numpy array, float
        Array of input values which standard deviation is requested.
    LDEBUG : bool
        If True prints values.

    Returns
    -------
    sigmag : float
        Robust estimator of the standar deviation
    """

    q25, q75 = np.percentile(x, [25.0, 75.0])
    sigmag = 0.7413 * (q75 - q25)

    return sigmag


# ------------------------------------------------------------------------------

def statsummary(x, LDEBUG=False):
    """Compute basic statistical parameters

    Parameters
    ----------
    x : 1d numpy array, float
        Input array with values which statistical properties are requested.
    LDEBUG : bool
        If True prints values.

    Returns
    -------
    result : tuple, floats
        Minimum, percentile 25, percentile 50, mean, percentile 75, maximum,
        standard deviation, and robust standard deviation.

    """

    result = np.min(x), \
             np.percentile(x, 25), \
             np.percentile(x, 50), \
             np.mean(x), \
             np.percentile(x, 75), \
             np.max(x), \
             np.std(x), \
             sigmaG(x), \
             np.percentile(x, 15.86553), \
             np.percentile(x, 84.13447)

    if LDEBUG:
        print('>>> Minimum..................:', result[0])
        print('>>> 1st Quartile.............:', result[1])
        print('>>> Median...................:', result[2])
        print('>>> Mean.....................:', result[3])
        print('>>> 3rd Quartile.............:', result[4])
        print('>>> Maximum..................:', result[5])
        print('>>> Standard deviation.......:', result[6])
        print('>>> Robust standard deviation:', result[7])
        print('>>> 0.1586553 percentile.....:', result[8])
        print('>>> 0.8413447 percentile.....:', result[9])

    return result

# ------------------------------------------------------------------------------
