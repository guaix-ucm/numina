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

from __future__ import division

from scipy.stats import norm
import numpy as np
from astropy.modeling.models import custom_model_1d
import math

M_SQRT_2_PI = math.sqrt(2 * math.pi)


def norm_pdf_t(x):
    return np.exp(-0.5 * x * x) / M_SQRT_2_PI


def gauss_box_model(x, amplitude=1.0, location=0.0, s=1.0, d=0.5):
    '''Integrate a Gaussian profile.'''
    m2 = (x + d - location) / s
    m1 = (x - d - location) / s
    return amplitude * (norm.cdf(m2) - norm.cdf(m1))


def gauss_box_model_deriv(x, amplitude=1.0, location=0.0, s=1.0, d=0.5):
    '''Integrate a Gaussian profile.'''
    z2 = (x + d - location) / s
    z1 = (x - d - location) / s

    da = norm.cdf(z2) - norm.cdf(z1)

    fp2 = norm_pdf_t(z2)
    fp1 = norm_pdf_t(z1)

    dl = -amplitude / s * (fp2 - fp1)
    ds = -amplitude / s * (fp2 * z2 - fp1 * z1)
    dd = amplitude / s * (fp2 + fp1)

    return (da, dl, ds, dd)


GaussBox = custom_model_1d(
    gauss_box_model, func_fit_deriv=gauss_box_model_deriv)
