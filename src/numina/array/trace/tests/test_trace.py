#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..traces import trace
from ..traces import axis_to_dispaxis
from ..traces import tracing_limits


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
    """Trace doesn't work with a flat peak"""
    arr = np.zeros((100, 100))
    arr[47:52, 12:90] = 100.0
    mm = trace(arr, 50, 50)
    assert mm.shape[0] >= 1


def helper_lim_min(col, step, hs):
    while col > hs + step:
        col += -1 * step
    return col


def helper_lim_max(col, step, hs, size):
    while col + step + hs < size:
        col += step
    return col


@pytest.mark.parametrize("size", [4096, 4097])
@pytest.mark.parametrize("col", list(range(1998, 2003)))
@pytest.mark.parametrize("step", [1,2,3])
@pytest.mark.parametrize("hs", [1,2,3])
def test_lower_limit(size, col, step,hs):
    vcalc_min, vcalc_max = tracing_limits(size, col, step, hs)

    vreal_min = helper_lim_min(col, step, hs)
    vreal_max = helper_lim_max(col, step, hs, size)

    assert vcalc_min == vreal_min
    assert vcalc_max == vreal_max
