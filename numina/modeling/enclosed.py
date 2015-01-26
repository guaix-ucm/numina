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

import numpy as np
from astropy.modeling import FittableModel, Parameter, format_input


class EnclosedGaussian(FittableModel):
    '''Enclosed Gaussian model'''
    amplitude = Parameter()
    stddev = Parameter()

    def __init__(self, amplitude, stddev, **kwargs):
        super(EnclosedGaussian, self).__init__(
            amplitude=amplitude, stddev=stddev, **kwargs)

    @staticmethod
    def eval(x, amplitude, stddev):
        return amplitude * (1 - np.exp(-0.5 * (x / stddev)**2))

    @staticmethod
    def fit_deriv(x, amplitude, stddev):
        z = (x / stddev)**2
        t = np.exp(-0.5 * z)
        d_amplitude = -t + 1.0
        d_stddev = -amplitude * t * z / stddev
        return [d_amplitude, d_stddev]

    @format_input
    def __call__(self, x):
        return self.eval(x, *self.param_sets)
