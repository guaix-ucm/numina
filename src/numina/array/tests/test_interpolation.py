#
# Copyright 2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

import numpy as np

from ..interpolation import SteffenInterpolator


def test_extrapolation_raise_by_default():
    xi = np.arange(0, 10, 0.5)
    yi = 3 * xi

    stfi = SteffenInterpolator(xi, yi)
    with pytest.raises(ValueError):
        stfi(200)

    stfi1 = SteffenInterpolator(xi, yi, extrapolate='raise')
    with pytest.raises(ValueError):
        stfi1(200)


def test_extrapolation_check_unknown_mode():
    xi = np.arange(0, 10, 0.5)
    yi = 3 * xi

    with pytest.raises(ValueError):
        SteffenInterpolator(xi, yi, extrapolate='kisjd')


def test_extrapolation_check_known_mode():
    xi = np.arange(0, 10, 0.5)
    yi = 3 * xi

    m = SteffenInterpolator(xi, yi, extrapolate='raise')
    assert isinstance(m, SteffenInterpolator)

    m = SteffenInterpolator(xi, yi, extrapolate='zeros')
    assert isinstance(m, SteffenInterpolator)

    m = SteffenInterpolator(xi, yi, extrapolate='const')
    assert isinstance(m, SteffenInterpolator)

    m = SteffenInterpolator(xi, yi, extrapolate='border')
    assert isinstance(m, SteffenInterpolator)

    m = SteffenInterpolator(xi, yi, extrapolate='extrapolate')
    assert isinstance(m, SteffenInterpolator)


def test_interpolation():

    def function(x):
        return x * (12*np.sin(x)-3)-0.3*x**2*(8*np.cos(3.1*x)-3) / (x+23.4)

    xi = np.arange(0, 6.5, 0.5)
    yi = function(xi)

    sti = SteffenInterpolator(xi, yi)
    xnew = np.arange(0, 6.5, 0.5)
    assert np.allclose(sti(xnew), yi)

    xnew = np.linspace(1.2, 1.4, 10)

    expected = np.array([9.94082135,  10.2539533,  10.56475383,  10.87191933, 11.1741462,
                         11.47013086, 11.7585697, 12.03815913, 12.30759555,  12.56557537])
    result = sti(xnew)

    assert np.allclose(result, expected)


def test_extrapolation_zeros():

    def function(x):
        return x * (12*np.sin(x)-3)-0.3*x**2*(8*np.cos(3.1*x)-3) / (x+23.4)

    xi = np.arange(0, 6.0)
    yi = 1.0+ 3 * xi

    sti = SteffenInterpolator(xi, yi, extrapolate='zeros')

    xnew = [6.0, 7.0]
    result = sti(xnew)
    expected = [0.0, 0.0]
    assert np.allclose(result, expected)

    xnew = [-1.0, -0.1]
    result = sti(xnew)
    expected = [0.0, 0.0]
    assert np.allclose(result, expected)


def test_extrapolation_const():

    xi = np.arange(0, 6.0)
    yi = 1.0 + 3 * xi

    fill = 12.0
    sti = SteffenInterpolator(xi, yi, extrapolate='const', fill_value=fill)

    xnew = [6.0, 7.0]
    result = sti(xnew)
    expected = [fill, fill]
    assert np.allclose(result, expected)

    xnew = [-1.0, -0.1]
    result = sti(xnew)
    expected = [fill, fill]
    assert np.allclose(result, expected)


def test_extrapolation_border():

    xi = np.arange(0, 6.0)
    yi = 1.0 + 3 * xi

    sti = SteffenInterpolator(xi, yi, extrapolate='border')

    xnew = [6.0, 7.0]
    result = sti(xnew)
    expected = [yi[-1], yi[-1]]
    assert np.allclose(result, expected)

    xnew = [-1.0, -0.1]
    result = sti(xnew)
    expected = [yi[0], yi[0]]
    assert np.allclose(result, expected)


def test_extrapolation_extrapolate():

    xi = np.arange(0, 6.0)
    yi = 1.0 + 3 * xi

    sti = SteffenInterpolator(xi, yi, extrapolate='extrapolate')

    xnew = [6.0, 7.0]
    result = sti(xnew)
    expected = [7, -32.0]
    assert np.allclose(result, expected)

    xnew = [-1.0, -0.1]
    result = sti(xnew)
    expected = [10.0, 1.063]
    assert np.allclose(result, expected)


def test_conditions_in_borders():

    def function(x):
        return x * (12*np.sin(x)-3)-0.3*x**2*(8*np.cos(3.1*x)-3) / (x+23.4)

    xi = np.arange(0, 6.0)
    yi = 1.0+ 3 * xi

    with pytest.raises(ValueError):
        SteffenInterpolator(xi, yi, yp_0=200)

    with pytest.raises(ValueError):
        SteffenInterpolator(xi, yi, yp_0=200)

    with pytest.raises(ValueError):
        SteffenInterpolator(xi, yi, yp_N=200)

    with pytest.raises(ValueError):
        SteffenInterpolator(xi, yi, yp_N=-200)
