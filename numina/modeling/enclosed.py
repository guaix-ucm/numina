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

