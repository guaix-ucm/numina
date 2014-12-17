#
# Copyright 2014 Universidad Complutense de Madrid
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

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ..utils import wc_to_pix_1d, wcs_to_pix_np

def test_wctopix():

    xin = [0, 1, 2, 3, 4, 5]
    cpix = [wc_to_pix_1d(x) for x in xin]
    pix = [0, 1, 2, 3, 4, 5]

    assert_array_equal(cpix, pix)
    
    xin = [0.1, 1.6, 2.8, 3.5, 3.500001, 4.9999, 5.7]
    cpix = [wc_to_pix_1d(x) for x in xin]
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
    pass


def test_expand_slice():
    pass


def test_expand_region():
    pass
