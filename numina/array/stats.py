#
# Copyright 2015-2016 Universidad Complutense de Madrid
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

import numpy
import numpy as np


def robust_std(x, debug=False):
    """Compute a robust estimator of the standard deviation

    See Eq. 3.36 (page 84) in Statistics, Data Mining, and Machine
    in Astronomy, by Ivezic, Connolly, VanderPlas & Gray

    Parameters
    ----------
    x : 1d numpy array, float
        Array of input values which standard deviation is requested.
    debug : bool
        If True prints computed values

    Returns
    -------
    sigmag : float
        Robust estimator of the standar deviation
    """

    x = numpy.asarray(x)

    # compute percentiles and robust estimator
    q25 = numpy.percentile(x, 25)
    q75 = numpy.percentile(x, 75)
    sigmag = 0.7413 * (q75 - q25)

    if debug:
        print('debug|sigmag -> q25......................:', q25)
        print('debug|sigmag -> q75......................:', q75)
        print('debug|sigmag -> Robust standard deviation:', sigmag)

    return sigmag


def summary(x, rm_nan=False, debug=False):
    """Compute basic statistical parameters.

    Parameters
    ----------
    x : 1d numpy array, float
        Input array with values which statistical properties are 
        requested.
    rm_nan : bool
        If True, filter out NaN values before computing statistics.
    debug : bool
        If True prints computed values.

    Returns
    -------
    result : Python dictionary
        Number of points, minimum, percentile 25, percentile 50
        (median), mean, percentile 75, maximum, standard deviation,
        robust standard deviation, percentile 15.866 (equivalent
        to -1 sigma in a normal distribution) and percentile 84.134
        (+1 sigma).

    """

    # protections
    if type(x) is not np.ndarray:
        raise ValueError('x=' + str(x) + ' must be a numpy.ndarray')

    if x.ndim is not 1:
        raise ValueError('x.dim=' + str(x.ndim) + ' must be 1')

    # filter out NaN's
    xx = np.copy(x)
    if rm_nan:
        xx = xx[np.logical_not(np.isnan(xx))]

    # compute basic statistics
    npoints = len(xx)
    ok = npoints > 0
    result = {
        'npoints' : npoints,
        'minimum' : np.min(xx) if ok else 0,
        'percentile25' : np.percentile(xx, 25) if ok else 0,
        'median' : np.percentile(xx, 50) if ok else 0,
        'mean' : np.mean(xx) if ok else 0,
        'percentile75': np.percentile(xx, 75) if ok else 0,
        'maximum' : np.max(xx) if ok else 0,
        'std': np.std(xx) if ok else 0,
        'robust_std' : robust_std(xx) if ok else 0,
        'percentile15': np.percentile(xx, 15.86553) if ok else 0,
        'percentile84': np.percentile(xx, 84.13447) if ok else 0
    }

    if debug:
        print('>>> ========================================')
        print('>>> STATISTICAL SUMMARY:')
        print('>>> ----------------------------------------')
        print('>>> Number of points.........:', result['npoints'])
        print('>>> Minimum..................:', result['minimum'])
        print('>>> 1st Quartile.............:', result['percentile25'])
        print('>>> Median...................:', result['median'])
        print('>>> Mean.....................:', result['mean'])
        print('>>> 3rd Quartile.............:', result['percentile75'])
        print('>>> Maximum..................:', result['maximum'])
        print('>>> ----------------------------------------')
        print('>>> Standard deviation.......:', result['std'])
        print('>>> Robust standard deviation:', result['robust_std'])
        print('>>> 0.1586553 percentile.....:', result['percentile15'])
        print('>>> 0.8413447 percentile.....:', result['percentile84'])
        print('>>> ========================================')

    return result

