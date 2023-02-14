#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Unit test for extract"""

import numpy as np
from numpy.testing import assert_allclose

from  ..extract import extract_simple_rss_apers
from  ..extract import Aperture
from  ..traces import axis_to_dispaxis


def test_extract_simple_rss_apers_flat():

    img = np.ones((89, 100))

    bbox = [5, 71, 6, 64]
    # Check with flat boundaries
    # axis = 0
    aa1 = 2.0
    aa2 = 5.5
    axis = 0
    dispaxis = axis_to_dispaxis(axis)
    b1 = lambda x: aa1 * np.ones_like(x)
    b2 = lambda x: aa2 * np.ones_like(x)

    result = np.zeros((1, img.shape[1]))
    result[0, bbox[0]:bbox[1]+1] = aa2 - aa1

    aper = Aperture(bbox, [b1,b2], axis=0)
    out = extract_simple_rss_apers(img, [aper], axis=axis)

    assert out.shape[1] ==  img.shape[dispaxis]
    assert_allclose(out, result)

    # Check with flat boundaries
    # axis = 1
    aa3 = 1.9
    aa4 = 5.5
    axis = 1
    dispaxis = axis_to_dispaxis(axis)
    b3 = lambda x: aa3 * np.ones_like(x)
    b4 = lambda x: aa4 * np.ones_like(x)

    result = np.zeros((1, img.shape[0]))
    result[0, bbox[2]:bbox[3]+1] = aa4 - aa3

    aper = Aperture(bbox, [b3,b4], axis=1)
    out = extract_simple_rss_apers(img, [aper], axis=axis)
    assert out.shape[1] ==  img.shape[dispaxis]

    assert_allclose(out, result)

def test_extract_simple_rss_apers_line():

    img = np.ones((89, 100))
    bbox = [5, 71, 6, 64]

    # Check with flat boundaries
    # axis = 0
    aa1 = 2.0
    aa2 = 5.5

    axis = 0
    dispaxis = axis_to_dispaxis(axis)
    b1 = lambda x: aa1 + 1.1 * (x - 50) / 100
    b2 = lambda x: aa2 + 1.1 * (x - 50) / 100

    result = np.zeros((1, img.shape[1]))
    result[0, bbox[0]:bbox[1]+1] = aa2 - aa1

    aper = Aperture(bbox, [b1, b2], axis=0)
    out = extract_simple_rss_apers(img, [aper], axis=axis)

    assert out.shape[1] ==  img.shape[dispaxis]
    assert_allclose(out, result)

    # Check with flat boundaries
    # axis = 1
    aa3 = 1.9
    aa4 = 5.5

    b3 = lambda x: aa3 + 1.2 * (x - 44) / 89
    b4 = lambda x: aa4 + 1.2 * (x - 44) / 89
    axis = 1
    dispaxis = axis_to_dispaxis(axis)

    result = np.zeros((1, img.shape[0]))
    result[0, bbox[2]:bbox[3]+1] = aa4 - aa3

    aper = Aperture(bbox, [b3, b4], axis=1)
    out = extract_simple_rss_apers(img, [aper], axis=axis)

    assert out.shape[1] ==  img.shape[dispaxis]
    assert_allclose(out, result)
