#
# Copyright 2014-2017 Universidad Complutense de Madrid
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

from __future__ import division

from scipy.stats import norm
import numpy as np
from astropy.modeling.models import custom_model
import math

M_SQRT_2_PI = math.sqrt(2 * math.pi)


def norm_pdf_t(x):
    return np.exp(-0.5 * x * x) / M_SQRT_2_PI


def gauss_box_model(x, amplitude=1.0, mean=0.0, stddev=1.0, hpix=0.5):
    """Integrate a Gaussian profile."""
    z = (x - mean) / stddev
    z2 = z + hpix / stddev
    z1 = z - hpix / stddev
    return amplitude * (norm.cdf(z2) - norm.cdf(z1))


def gauss_box_model_deriv(x, amplitude=1.0, mean=0.0, stddev=1.0, hpix=0.5):
    """Derivative of the integral of  a Gaussian profile."""
    z = (x - mean) / stddev
    z2 = z + hpix / stddev
    z1 = z - hpix / stddev

    da = norm.cdf(z2) - norm.cdf(z1)

    fp2 = norm_pdf_t(z2)
    fp1 = norm_pdf_t(z1)

    dl = -amplitude / stddev * (fp2 - fp1)
    ds = -amplitude / stddev * (fp2 * z2 - fp1 * z1)
    dd = amplitude / stddev * (fp2 + fp1)

    return da, dl, ds, dd


GaussBox = custom_model(
    gauss_box_model, func_fit_deriv=gauss_box_model_deriv)
