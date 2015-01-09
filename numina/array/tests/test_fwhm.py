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

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from astropy.modeling.models import Gaussian2D

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
    assert_allclose(fwhmx, 2.3548200450309493 * xsig, rtol=1e-3)
    assert_allclose(fwhmy, 2.3548200450309493 * ysig, rtol=1e-3)


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
    assert_allclose(fwhm, 2.3548200450309493 * sigma, rtol=1e-4)

    # Test with a gaussian, not starting in 0
    rad = np.arange(10, 260, 1.0)
    center = 130.2
    sigma = 25.3
    Fr_ref = np.exp(-0.5 * ((rad - center) / sigma)**2)

    peak, fwhm = compute_fwhm_1d_simple(Fr_ref, center, rad)
    assert_allclose(fwhm, 2.3548200450309493 * sigma, rtol=1e-4)
