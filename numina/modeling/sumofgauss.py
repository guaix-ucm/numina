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


def sum_of_gaussian_factory(N):
    '''Return a model of the sum of N Gaussians and a constant background.'''

    name = "SumNGauss%d" % N
    attr = {}

    # parameters
    for i in range(N):
        key = "amplitude_%d" % i
        attr[key] = Parameter(key)
        key = "center_%d" % i
        attr[key] = Parameter(key)
        key = "stddev_%d" % i
        attr[key] = Parameter(key)

    attr['background'] = Parameter('background', default=0.0)

    def fit_eval(self, x, *args):
        result = x * 0 + args[-1]
        for i in range(N):
            result += args[3 * i] * \
                np.exp(- 0.5 * (x - args[3 * i + 1])
                       ** 2 / args[3 * i + 2] ** 2)
        return result

    attr['eval'] = fit_eval

    def deriv(self, x, *args):
        d_result = np.ones(((3 * N + 1), len(x)))

        for i in range(N):
            d_result[3 * i] = (np.exp(-0.5 / args[3 * i + 2] ** 2 *
                                      (x - args[3 * i + 1]) ** 2))
            d_result[3 * i + 1] = args[3 * i] * d_result[3 * i] * \
                (x - args[3 * i + 1]) / args[3 * i + 2] ** 2
            d_result[3 * i + 2] = args[3 * i] * d_result[3 * i] * \
                (x - args[3 * i + 1]) ** 2 / args[3 * i + 2] ** 3
        return d_result

    attr['fit_deriv'] = deriv

    @format_input
    def __call__(self, x):
        return self.eval(x, *self.param_sets)

    attr['__call__'] = __call__

    klass = type(name, (FittableModel, ), attr)
    return klass
