#
# Copyright 2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


class EnclosedGaussian(Fittable1DModel):
    '''Enclosed Gaussian model'''

    amplitude = Parameter()
    stddev = Parameter()

    @staticmethod
    def evaluate(x, amplitude, stddev):
        return amplitude * (1 - np.exp(-0.5 * (x / stddev)**2))

    @staticmethod
    def fit_deriv(x, amplitude, stddev):
        z = (x / stddev)**2
        t = np.exp(-0.5 * z)
        d_amplitude = -t + 1.0
        d_stddev = -amplitude * t * z / stddev
        return [d_amplitude, d_stddev]

