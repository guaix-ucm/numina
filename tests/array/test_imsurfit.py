#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import pytest

import numpy

from numina.array.imsurfit import imsurfit, vertex_of_quadratic


def calc_ws(xx, yy):
    res = [
        1,
        xx,
        yy,
        xx**2,
        xx * yy,
        yy**2,
        xx**3,
        xx**2 * yy,
        xx * yy**2,
        yy**3,
        xx**4,
        xx**3 * yy,
        xx**2 * yy**2,
        xx * yy**3,
        yy**4,
    ]
    return res


@pytest.mark.parametrize("steps", [100, 458])
@pytest.mark.parametrize(
    "order, results",
    [
        (1, [456.0, 0.3, -0.9]),
        (2, [1.0, 0.1, 12, 3.0, -11.8, 9.2]),
        (3, [456.0, 0.3, -0.9, 0.03, -0.01, 0.07, 0.0, -10, 0.0, 0.04]),
        (4, [-11.0, -1.5, -0.1, 0.14, -15.03, 0.07, 0.448, -0.28, 1.4, 1.24, -3.2, -1.2, 2.24, -8.1, -0.03]),
    ],
)
def test_fit(order, results, steps):
    """Test cuartic fitting."""
    results0 = numpy.array(results)

    x0 = numpy.linspace(-1.0, 1.0, steps)
    y0 = numpy.linspace(-1.0, 1.0, steps)
    xx, yy = numpy.meshgrid(x0, y0, indexing="ij")
    # xx, yy = numpy.mgrid[-1:1:100j, -1:1:100j]
    ws0 = calc_ws(xx, yy)

    dd = [cf * v for cf, v in zip(results0.flat, ws0)]
    z = sum(dd)
    (results,) = imsurfit(z, order=order)
    assert numpy.allclose(results0, results)


def test_fit_output():
    x0 = numpy.linspace(-1.0, 1.0, 50)
    xx, yy = numpy.meshgrid(x0, x0, indexing="ij")
    z = 1.0 + 0.1 * xx + 12 * yy + 3.0 * xx**2 - 11.8 * xx * yy + 9.2 * yy**2
    coeffs, fit = imsurfit(z, order=2, output_fit=True)
    assert numpy.allclose(coeffs, [1.0, 0.1, 12, 3.0, -11.8, 9.2])
    assert numpy.allclose(fit, z)


def test_vertex_of_quadratic():
    # f = 0.4 x - 0.4 y - x**2 - 2 y**2, maximum in (0.2, -0.1)
    coeffs = [0.0, 0.4, -0.4, -1.0, 0.0, -2.0]
    assert numpy.allclose(vertex_of_quadratic(coeffs), (0.2, -0.1))


def test_vertex_of_quadratic_saddle():
    with pytest.raises(ValueError, match="no maximum"):
        vertex_of_quadratic([0.0, 0.0, 0.0, 1.0, 0.0, -1.0])


def test_vertex_of_fitted_surface():
    x0 = numpy.linspace(-1.0, 1.0, 21)
    xx, yy = numpy.meshgrid(x0, x0, indexing="ij")
    z = 5.0 - (xx - 0.3) ** 2 - 2 * (yy + 0.2) ** 2
    (coeffs,) = imsurfit(z, order=2)
    assert numpy.allclose(vertex_of_quadratic(coeffs), (0.3, -0.2))
