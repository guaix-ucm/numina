#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from scipy.stats import norm
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
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


class GaussBox(Fittable1DModel):
    """Model for fitting a 1D Gaussina convolved with a square"""
    amplitude = Parameter(default=1.0)
    mean = Parameter(default=0.0)
    stddev = Parameter(default=1.0)
    hpix = Parameter(default=0.5, fixed=True)

    @staticmethod
    def evaluate(x, amplitude, mean, stddev, hpix):
        return gauss_box_model(x, amplitude, mean, stddev, hpix)

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev, hpix):
        return gauss_box_model_deriv(x, amplitude, mean, stddev, hpix)

