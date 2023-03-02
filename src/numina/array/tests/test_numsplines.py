
import numpy as np
from numpy.testing import assert_allclose
from ..numsplines import AdaptiveLSQUnivariateSpline


def test_adaptive_sqlunivariatespline():
    npoints = 300
    np.random.seed(1234)
    xdata = np.linspace(-2*np.pi, 2*np.pi, npoints)
    ydata = np.sin(xdata) + 0.3 * np.random.randn(npoints)

    # adaptive splines
    spl = AdaptiveLSQUnivariateSpline(xdata, ydata, 2)
    xknots = spl.get_knots()
    yknots = spl(xknots)

    assert_allclose(
        spl.get_coeffs(),
        [-0.2450535, 3.15287065, -3.6363213, 3.60549501,
         -2.96073338,  0.18723241], rtol=1e-6)
    assert_allclose(spl.get_residual(), 26.538044076506353)
    assert_allclose(xknots,
                    [-6.28318531, -1.34353445, 1.30888191, 6.28318531],
                    rtol=1e-6)
    assert_allclose(yknots,
                    [-0.2450535, -0.95552754, 0.94163857, 0.18723241],
                    rtol=1e-6)

    # fixed knots
    spl = AdaptiveLSQUnivariateSpline(xdata, ydata, [-4, 0, 4], adaptive=False)
    result = spl.get_result()
    assert result is None
    xknots = spl.get_knots()
    yknots = spl(xknots)
    assert_allclose(
        spl.get_coeffs(),
        [-0.65286453, 2.38128699, -1.26391753, -0.05672272,
         1.37701644, -2.33129953, 0.61925078],
        rtol=1e-6
    )
    assert_allclose(spl.get_residual(), 59.696265273407434)
    assert_allclose(
        xknots, [-6.28318531, -4., 0., 4., 6.28318531], rtol=1e-6
    )
    assert_allclose(yknots, [-0.65286453, 0.31082669, -0.01266159,
                             -0.24158386, 0.61925078], rtol=1e-6)
