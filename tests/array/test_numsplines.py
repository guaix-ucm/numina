import numpy as np
from numpy.testing import assert_allclose
from lmfit import Parameters
import pytest
from numina.array.numsplines import AdaptiveLSQUnivariateSpline, _initial_simplex


def test_initial_simplex():
    params = Parameters()
    params.add("a", value=0.0, min=-1.0, max=1.0)  # centre of its interval
    params.add("b", value=0.9, min=0.5, max=1.0)  # close to the upper bound
    params.add("c", value=5.0, vary=False)
    simplex = _initial_simplex(params, fraction=0.1)
    assert simplex.shape == (3, 2)
    u0 = np.array([params["a"].setup_bounds(), params["b"].setup_bounds()])
    assert_allclose(simplex[0], u0)
    # each vertex displaces one parameter, towards its farthest bound,
    # by 10% of the width of its interval
    assert_allclose(simplex[1], [np.arcsin(0.2), u0[1]])
    assert_allclose(simplex[2], [u0[0], np.arcsin(2 * (0.85 - 0.5) / 0.5 - 1)])


@pytest.mark.parametrize("nknots", [3, 4, 5])
def test_adaptive_knots_independent_of_rounding(nknots):
    # see issue #285: knots starting at the centre of their interval
    # could remain frozen depending on the rounding of the last digit of x
    x0 = np.linspace(8000.0, 9700.0, 4300)
    y = 1000 + 300 * np.exp(-0.5 * ((x0 - 8700) / 400) ** 2) - 250 * np.exp(-0.5 * ((x0 - 9300) / 60) ** 2)
    ref = AdaptiveLSQUnivariateSpline(x0, y, t=nknots)
    for offset in (1e-12, 2e-12, 3e-12, 4e-12):
        spl = AdaptiveLSQUnivariateSpline(x0 + offset, y, t=nknots)
        assert_allclose(spl.get_knots(), ref.get_knots(), rtol=1e-6)
        assert_allclose(spl(x0), ref(x0), rtol=1e-5)
    if nknots == 3:
        # the middle knot must move away from its equidistant start (8850)
        assert abs(ref.get_knots()[2] - 8850.0) > 100


def test_adaptive_sqlunivariatespline():
    npoints = 300
    np.random.seed(1234)
    xdata = np.linspace(-2 * np.pi, 2 * np.pi, npoints)
    ydata = np.sin(xdata) + 0.3 * np.random.randn(npoints)

    # adaptive splines
    spl = AdaptiveLSQUnivariateSpline(xdata, ydata, 2)
    xknots = spl.get_knots()
    yknots = spl(xknots)

    assert_allclose(
        spl.get_coeffs(), [-0.2450535, 3.15287065, -3.6363213, 3.60549501, -2.96073338, 0.18723241], rtol=1e-6
    )
    assert_allclose(spl.get_residual(), 26.538044076506353)
    assert_allclose(xknots, [-6.28318531, -1.34353445, 1.30888191, 6.28318531], rtol=1e-6)
    assert_allclose(yknots, [-0.2450535, -0.95552754, 0.94163857, 0.18723241], rtol=1e-6)

    # fixed knots
    spl = AdaptiveLSQUnivariateSpline(xdata, ydata, [-4, 0, 4], adaptive=False)
    result = spl.get_result()
    assert result is None
    xknots = spl.get_knots()
    yknots = spl(xknots)
    assert_allclose(
        spl.get_coeffs(),
        [-0.65286453, 2.38128699, -1.26391753, -0.05672272, 1.37701644, -2.33129953, 0.61925078],
        rtol=1e-6,
    )
    assert_allclose(spl.get_residual(), 59.696265273407434)
    assert_allclose(xknots, [-6.28318531, -4.0, 0.0, 4.0, 6.28318531], rtol=1e-6)
    assert_allclose(yknots, [-0.65286453, 0.31082669, -0.01266159, -0.24158386, 0.61925078], rtol=1e-6)
