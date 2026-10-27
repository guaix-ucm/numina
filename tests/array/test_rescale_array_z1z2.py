#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Tests for numina.array.rescale_array_z1z2"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numina.array.rescale_array_z1z2 import rescale_array_to_z1z2
from numina.array.rescale_array_z1z2 import rescale_array_from_z1z2


@pytest.mark.parametrize("z1z2", [(-1.0, 1.0), (0.0, 1.0), (10.0, 20.0), (1.0, -1.0)])
def test_rescale_to_z1z2_limits(z1z2):
    """Minimum and maximum of the input are mapped to z1 and z2."""
    array = np.array([3.0, 7.0, 5.0, 11.0, 4.0])
    z1, z2 = z1z2
    array_rs, _ = rescale_array_to_z1z2(array, z1z2)
    assert_allclose(array_rs[0], z1)
    assert_allclose(array_rs[3], z2)


def test_rescale_to_z1z2_coefficients():
    """Coefficients follow Appendix B1 of Cardiel (2009)."""
    array = np.array([2.0, 4.0, 10.0])
    array_rs, (b_flux, c_flux) = rescale_array_to_z1z2(array, (-1.0, 1.0))
    # b = (z2 - z1) / (max - min), c = (z2 * min - z1 * max) / (max - min)
    assert_allclose(b_flux, 2.0 / 8.0)
    assert_allclose(c_flux, (2.0 + 10.0) / 8.0)
    assert_allclose(array_rs, [-1.0, -0.5, 1.0])


def test_rescale_to_z1z2_integer_input():
    """Integer arrays are rescaled to floats."""
    array = np.array([1, 2, 3])
    array_rs, _ = rescale_array_to_z1z2(array)
    assert_allclose(array_rs, [-1.0, 0.0, 1.0])


def test_rescale_to_z1z2_2d():
    """Multidimensional arrays keep their shape."""
    array = np.arange(12, dtype=float).reshape(3, 4)
    array_rs, _ = rescale_array_to_z1z2(array, (0.0, 1.0))
    assert array_rs.shape == array.shape
    assert_allclose(array_rs, array / 11.0)


@pytest.mark.parametrize("badvalue", [np.nan, np.inf, -np.inf])
def test_rescale_to_z1z2_nonfinite(badvalue):
    """Non-finite values are ignored in the coefficients and preserved."""
    array = np.array([1.0, badvalue, 3.0, 5.0])
    array_rs, (b_flux, c_flux) = rescale_array_to_z1z2(array)
    assert np.isfinite(b_flux)
    assert np.isfinite(c_flux)
    assert_allclose(array_rs[[0, 2, 3]], [-1.0, 0.0, 1.0])
    assert_array_equal(array_rs[1], badvalue)


@pytest.mark.parametrize(
    "array",
    [
        np.array([np.nan, np.nan]),
        np.array([np.inf, -np.inf, np.nan]),
    ],
)
def test_rescale_to_z1z2_no_finite_values(array):
    """An array without finite values raises ValueError."""
    with pytest.raises(ValueError, match="finite"):
        rescale_array_to_z1z2(array)


def test_rescale_to_z1z2_constant_array():
    """A constant array cannot be rescaled."""
    with pytest.raises(ValueError):
        rescale_array_to_z1z2(np.array([2.0, 2.0, np.nan]))


@pytest.mark.parametrize("func", [rescale_array_to_z1z2, rescale_array_from_z1z2])
def test_rescale_not_ndarray(func):
    """Inputs that are not numpy.ndarray raise ValueError."""
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        if func is rescale_array_to_z1z2:
            func([1.0, 2.0, 3.0])
        else:
            func([1.0, 2.0, 3.0], (1.0, 0.0))


@pytest.mark.parametrize("z1z2", [(-1.0, 1.0), (0.0, 1.0), (1.0, -1.0)])
def test_rescale_round_trip(z1z2):
    """rescale_array_from_z1z2 inverts rescale_array_to_z1z2."""
    rng = np.random.default_rng(1234)
    array = rng.normal(loc=1000.0, scale=50.0, size=(20, 30))
    array_rs, coef_rs = rescale_array_to_z1z2(array, z1z2)
    assert_allclose(rescale_array_from_z1z2(array_rs, coef_rs), array)


def test_rescale_round_trip_nonfinite():
    """The round trip preserves NaN and +/-inf at their positions."""
    array = np.array([1.0, np.nan, 3.0, np.inf, 5.0, -np.inf])
    array_rs, coef_rs = rescale_array_to_z1z2(array)
    assert_array_equal(rescale_array_from_z1z2(array_rs, coef_rs), array)


def test_rescale_from_z1z2_requires_coefficients():
    """coef_rs is a required argument."""
    with pytest.raises(TypeError):
        rescale_array_from_z1z2(np.array([0.0, 1.0]))
