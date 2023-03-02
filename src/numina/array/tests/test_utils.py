#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

import numpy as np
from numpy.testing import assert_array_equal

from ..utils import coor_to_pix_1d, wcs_to_pix_np
from ..utils import slice_create, expand_slice, expand_region


def test_wctopix():

    xin = [0, 1, 2, 3, 4, 5]
    cpix = [coor_to_pix_1d(x) for x in xin]
    pix = [0, 1, 2, 3, 4, 5]

    assert_array_equal(cpix, pix)

    xin = [0.1, 1.6, 2.8, 3.5, 3.500001, 4.9999, 5.7]
    cpix = [coor_to_pix_1d(x) for x in xin]
    pix = [0, 2, 3, 4, 4, 5, 6]

    assert_array_equal(cpix, pix)


@pytest.mark.skipif(reason='Not clear if this is the interface')
def test_wctopixnp():

    xin = np.array([[0, 1], [2, 3], [4, 5]])
    cpix = wcs_to_pix_np(xin)
    pix = np.array([[0, 1], [2, 3], [4, 5]])

    assert_array_equal(cpix, pix)

    xin = [0.1, 1.6, 2.8, 3.5, 3.500001, 4.9999, 5.7]
    cpix = wcs_to_pix_np(xin)
    pix = [0, 2, 3, 4, 4, 5, 6]

    assert_array_equal(cpix, pix)


def test_slice_create():

    val = slice(97, 104, 1)
    cval = slice_create(100, 3)
    assert val == cval

    # check limits
    cval = slice_create(10, 12)
    resval = slice(0, 23, 1)
    assert resval == cval

    cval = slice_create(100, 12, stop=105)
    resval = slice(88, 105, 1)
    assert resval == cval


def test_expand_slice():
    s = slice(94, 104, 1)
    cval = expand_slice(s, 3, 4)
    assert slice(91, 108, 1) == cval

    s = slice(94, 104, 1)
    cval = expand_slice(s, 3, 4, stop=106)
    assert slice(91, 106, 1) == cval

    s = slice(3, 100, 1)
    cval = expand_slice(s, 6, 5)
    assert slice(0, 105, 1) == cval


def test_expand_region():
    t = (slice(94, 104, 1), slice(103, 250, 1))
    ct = expand_region(t, 4, 5)
    assert ct == (slice(90, 109, 1), slice(99, 255, 1))
