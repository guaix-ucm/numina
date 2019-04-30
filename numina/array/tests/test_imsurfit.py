#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

import numpy

from numina.array.imsurfit import imsurfit


def calc_ws(xx, yy):
    res = [
        1, xx, yy, xx ** 2, xx * yy, yy ** 2,
        xx ** 3, xx ** 2 * yy, xx * yy ** 2, yy ** 3,
        xx ** 4, xx ** 3 * yy, xx ** 2 * yy ** 2,
        xx * yy ** 3, yy ** 4
    ]
    return res

@pytest.mark.parametrize("steps", [100, 458])
@pytest.mark.parametrize("order, results", [
    (1, [456.0, 0.3, -0.9]),
    (2, [1.0, 0.1, 12, 3.0, -11.8, 9.2]),
    (3, [456.0, 0.3, -0.9, 0.03, -0.01, 0.07, 0.0, -10, 0.0, 0.04]),
    (4, [-11.0, -1.5, -0.1, 0.14, -15.03,
         0.07,0.448, -0.28, 1.4, 1.24,
         -3.2, -1.2, 2.24, -8.1, -0.03])
])
def test_fit(order, results, steps):
    """Test cuartic fitting."""
    results0 = numpy.array(results)

    x0 = numpy.linspace(-1.0, 1.0, steps)
    y0 = numpy.linspace(-1.0, 1.0, steps)
    xx, yy = numpy.meshgrid(x0, y0, indexing='ij')
    # xx, yy = numpy.mgrid[-1:1:100j, -1:1:100j]
    ws0 = calc_ws(xx, yy)

    dd = [cf * v for cf, v in zip(results0.flat, ws0)]
    z = sum(dd)
    results, = imsurfit(z, order=order)
    assert numpy.allclose(results0, results)

