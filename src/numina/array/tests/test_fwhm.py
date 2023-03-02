#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from astropy.modeling.models import Gaussian2D

from ...constants import FWHM_G
from ..fwhm import compute_fwhm_2d_simple
from ..fwhm import compute_fwhm_1d_simple


def test_fwhm_2d_simple():
    xcenter = 120.2
    ycenter = 122.3
    xsig = 25.2
    ysig = 12.8
    g2d_model = Gaussian2D(amplitude=1.0, x_mean=xcenter,
                           y_mean=ycenter, x_stddev=xsig, y_stddev=ysig)
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
    Fr_ref = np.exp(-0.5 * ((rad - center) / sigma)**2)

    peak, fwhm = compute_fwhm_1d_simple(Fr_ref, center)
    assert_allclose(fwhm, FWHM_G * sigma, rtol=1e-4)

    # Test with a gaussian, not starting in 0
    rad = np.arange(10, 260, 1.0)
    center = 130.2
    sigma = 25.3
    Fr_ref = np.exp(-0.5 * ((rad - center) / sigma)**2)

    peak, fwhm = compute_fwhm_1d_simple(Fr_ref, center, rad)
    assert_allclose(fwhm, FWHM_G * sigma, rtol=1e-4)
