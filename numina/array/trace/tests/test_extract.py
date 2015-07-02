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

'''Unit test for extract'''

from __future__ import division

import numpy as np
from numpy.testing import assert_allclose

from  ..extract import extract_simple_rss_apers
from  ..extract import Aperture


def test_extract_simple_rss_apers_flat():

    img = np.ones((89, 100))

    bbox = [5, 71, 6, 64]
    # Check with flat boundaries
    # axis = 0
    aa1 = 2.0
    aa2 = 5.5

    b1 = lambda x: aa1 * np.ones_like(x)
    b2 = lambda x: aa2 * np.ones_like(x)

    result = np.zeros((1, img.shape[1]))
    result[0, bbox[2]:bbox[3]+1] = aa2 - aa1

    aper = Aperture(bbox, [b1,b2], axis=0)
    out = extract_simple_rss_apers(img, [aper])

    assert out.shape[1] ==  img.shape[1]
    assert out[0, bbox[0]] ==  aa2 - aa1
    assert out[0, bbox[1]] ==  aa2 - aa1

    # Check with flat boundaries
    # axis = 1
    aa3 = 1.9
    aa4 = 5.5

    b3 = lambda x: aa3 * np.ones_like(x)
    b4 = lambda x: aa4 * np.ones_like(x)

    result = np.zeros((1, img.shape[0]))
    result[0, bbox[2]:bbox[3]+1] = aa4 - aa3

    aper = Aperture(bbox, [b3,b4], axis=1)
    out = extract_simple_rss_apers(img, [aper], axis=1)
    assert out.shape[1] ==  img.shape[0]

    assert_allclose(out, result)

def test_extract_simple_rss_apers_line():

    img = np.ones((89, 100))
    bbox = [5, 71, 6, 64]

    # Check with flat boundaries
    # axis = 0
    aa1 = 2.0
    aa2 = 5.5

    b1 = lambda x: aa1 + 1.1 * (x - 50) / 100
    b2 = lambda x: aa2 + 1.1 * (x - 50) / 100

    result = np.zeros((1, img.shape[1]))
    result[0, bbox[0]:bbox[1]+1] = aa2 - aa1

    aper = Aperture(bbox, [b1, b2], axis=0)
    out = extract_simple_rss_apers(img, [aper])

    assert out.shape[1] ==  img.shape[1]
    assert_allclose(out, result)

    # Check with flat boundaries
    # axis = 1
    aa3 = 1.9
    aa4 = 5.5

    b3 = lambda x: aa3 + 1.2 * (x - 44) / 89
    b4 = lambda x: aa4 + 1.2 * (x - 44) / 89

    result = np.zeros((1, img.shape[0]))
    result[0, bbox[2]:bbox[3]+1] = aa4 - aa3

    aper = Aperture(bbox, [b3, b4], axis=1)
    out = extract_simple_rss_apers(img, [aper], axis=1)

    assert out.shape[1] ==  img.shape[0]
    assert_allclose(out, result)
