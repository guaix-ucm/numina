#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose
from astropy.modeling.models import Gaussian2D

from numina.constants import FWHM_G
from numina.array.fwhm import compute_fwhm_2d_simple, compute_fwhm_2d_spline
from numina.array.fwhm import compute_fwhm_1d_simple


def test_fwhm_2d_simple():
    xcenter = 120.2
    ycenter = 122.3
    xsig = 25.2
    ysig = 12.8
    g2d_model = Gaussian2D(amplitude=1.0, x_mean=xcenter, y_mean=ycenter, x_stddev=xsig, y_stddev=ysig)
    y, x = np.mgrid[:250, :250]
    img = g2d_model(x, y)
    _peak, fwhmx, fwhmy = compute_fwhm_2d_simple(img, xcenter, ycenter)
    assert_allclose(fwhmx, FWHM_G * xsig, rtol=1e-3)
    assert_allclose(fwhmy, FWHM_G * ysig, rtol=1e-3)


def test_fwhm_1d_simple():
    # Test with square box
    ref_peak = 1.0
    ref_val = 6.0

    Fr_ref = np.zeros((15,))
    Fr_ref[4:9] = ref_peak

    peak, fwhm = compute_fwhm_1d_simple(Fr_ref, ref_val)

    assert_almost_equal(peak, ref_peak)
    assert_allclose(fwhm, 5.0)

    # Test with a gaussian
    rad = np.arange(0, 250, 1.0)
    center = 120.2
    sigma = 25.3
    Fr_ref = np.exp(-0.5 * ((rad - center) / sigma) ** 2)

    peak, fwhm = compute_fwhm_1d_simple(Fr_ref, center)
    assert_allclose(fwhm, FWHM_G * sigma, rtol=1e-4)

    # Test with a gaussian, not starting in 0
    rad = np.arange(10, 260, 1.0)
    center = 130.2
    sigma = 25.3
    Fr_ref = np.exp(-0.5 * ((rad - center) / sigma) ** 2)

    peak, fwhm = compute_fwhm_1d_simple(Fr_ref, center, rad)
    assert_allclose(fwhm, FWHM_G * sigma, rtol=1e-4)


def test_fwhm_2d_spline():
    xcenter = 30.2
    ycenter = 20.7
    xsig = 6.0
    ysig = 3.0
    g2d_model = Gaussian2D(amplitude=1.0, x_mean=xcenter, y_mean=ycenter, x_stddev=xsig, y_stddev=ysig)
    y, x = np.mgrid[:50, :70]
    img = g2d_model(x, y)
    peak, fwhmx, fwhmy = compute_fwhm_2d_spline(img, xcenter, ycenter)
    assert_allclose(peak, 1.0, rtol=1e-3)
    assert_allclose(fwhmx, FWHM_G * xsig, rtol=1e-2)
    assert_allclose(fwhmy, FWHM_G * ysig, rtol=1e-2)


def test_fwhm_1d_one_side():
    # The profile reaches half the peak only on one side
    yy = np.array([1.0, 0.9, 0.8, 0.4, 0.2])
    _, fwhm = compute_fwhm_1d_simple(yy, 0)
    assert_allclose(fwhm, 2 * (2 + 0.3 / 0.4))
    _, fwhm = compute_fwhm_1d_simple(yy[::-1], 4)
    assert_allclose(fwhm, 2 * (2 + 0.3 / 0.4))


def test_fwhm_1d_failed():
    _, fwhm = compute_fwhm_1d_simple(np.ones(5), 2)
    assert fwhm == -99


@pytest.mark.parametrize(
    "yy, xc, msg",
    [
        (np.ones((3, 3)), 1, "must be 1-d"),
        (np.ones(0), 0, "is empty"),
        (np.ones(5), 7, "out of array"),
    ],
)
def test_fwhm_1d_invalid(yy, xc, msg):
    with pytest.raises(ValueError, match=msg):
        compute_fwhm_1d_simple(yy, xc)
