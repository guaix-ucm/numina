#
# Copyright 2014-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


def sum_of_gaussian_factory(N):
    """Return a model of the sum of N Gaussians and a constant background."""

    name = f"SumNGauss{N:d}"
    attr = {}

    # parameters
    for i in range(N):
        key = f"amplitude_{i:d}"
        attr[key] = Parameter(key)
        key = f"center_{i:d}"
        attr[key] = Parameter(key)
        key = f"stddev_{i:d}"
        attr[key] = Parameter(key)

    attr['background'] = Parameter('background', default=0.0)

    def fit_eval(self, x, *args):
        result = x * 0 + args[-1]
        for i in range(N):
            result += args[3 * i] * \
                np.exp(- 0.5 * (x - args[3 * i + 1])
                       ** 2 / args[3 * i + 2] ** 2)
        return result

    attr['evaluate'] = fit_eval

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

    klass = type(name, (Fittable1DModel, ), attr)
    return klass
