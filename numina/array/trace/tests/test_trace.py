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

'''Unit test for trace'''

from __future__ import division

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..traces import trace
from ..traces import axis_to_dispaxis


def test_axis_to_dispaxis():
    assert axis_to_dispaxis(0) == 1
    assert axis_to_dispaxis(1) == 0

    with pytest.raises(ValueError):
        axis_to_dispaxis(2)


@pytest.mark.parametrize("gauss", [1, 0])
def test_trace_simple(benchmark, gauss):
    '''Test a simple trace'''
    arr = np.zeros((100, 100))

    arr[47, 45:55] = 10.0
    arr[48, 45:55] = 100.0
    arr[49, 45:55] = 12.0

    if gauss:
        result = np.array(
            [[44., 48.02061133, 33.36466379], [45., 48.02061133, 66.72932758],
             [46., 48.02061133, 100.09399137], [47., 48.02061133, 100.09399137],
             [48., 48., 100.], [49., 48.02061133, 100.09399137],
             [50., 48.02061133, 100.09399137], [51., 48.02061133, 100.09399137],
             [52., 48.02061133, 100.09399137], [53., 48.02061133, 100.09399137],
             [54., 48.02061133, 66.72932758], [55., 48.02061133, 33.36466379]])
    else:
        result = np.empty((12, 3))
        result[:, 0] = np.arange(44, 56)
        result[:, 1] = 48.0056179775
        result[4, 1] = 48.0
        result[:, 2] = 100.00280898876404
        result[[0, 11], 2] = 33.33426966
        result[[1, 10], 2] = 66.66853933
        result[4, 2] = 100.0

    mm = benchmark(trace,arr, 48.0, 48.0, step=1, gauss=gauss)

    assert mm.shape == (12, 3)
    assert_allclose(mm, result)


def test_trace_bug_27():
    '''Trace doesn't work with a flat peak'''
    arr = np.zeros((100, 100))
    arr[47:52, 12:90] = 100.0
    mm = trace(arr, 50, 50)
    assert mm.shape[0] >= 1

# test_trace_simple[0]     15,974.0448 (80.72)      153,064.7278 (225.26)      18,480.6449 (87.36)      5,589.5218 (153.01)      16,927.7191 (84.52)       238.4186 (inf)       3065;12587   33289   1
# test_trace_simple[1]     16,927.7191 (85.54)      563,144.6838 (828.77)      20,605.2463 (97.40)      8,520.8203 (233.25)      18,119.8120 (90.48)     1,192.0929 (inf)        1556;2951   18478   1

# test_trace_simple[1]     14,781.9519 (78.48)      133,991.2415 (194.46)      16,926.2203 (81.15)     3,305.2872 (94.41)       15,974.0448 (79.76)       953.6743 (inf)         904;1148   20361    1
# test_trace_simple[0]     15,020.3705 (79.75)      142,097.4731 (206.23)      17,072.0888 (81.85)     3,499.1985 (99.95)       15,974.0448 (79.76)       953.6743 (inf)        1098;1416   23832    1

def test_trace_bug_27():
    '''Trace doesn't work with a flat peak'''
    arr = np.zeros((100, 100))
    arr[47:52, 12:90] = 100.0
    mm = trace(arr, 50, 50)
    assert mm.shape[0] >= 1

# test_trace_simple[0]     15,974.0448 (80.72)      153,064.7278 (225.26)      18,480.6449 (87.36)      5,589.5218 (153.01)      16,927.7191 (84.52)       238.4186 (inf)       3065;12587   33289   1
# test_trace_simple[1]     16,927.7191 (85.54)      563,144.6838 (828.77)      20,605.2463 (97.40)      8,520.8203 (233.25)      18,119.8120 (90.48)     1,192.0929 (inf)        1556;2951   18478   1

# test_trace_simple[1]     14,781.9519 (78.48)      133,991.2415 (194.46)      16,926.2203 (81.15)     3,305.2872 (94.41)       15,974.0448 (79.76)       953.6743 (inf)         904;1148   20361    1
# test_trace_simple[0]     15,020.3705 (79.75)      142,097.4731 (206.23)      17,072.0888 (81.85)     3,499.1985 (99.95)       15,974.0448 (79.76)       953.6743 (inf)        1098;1416   23832    1
