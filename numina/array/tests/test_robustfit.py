
import pytest
import numpy
from ..robustfit import fit_theil_sen


def test_theil_sen_except():

    with pytest.raises(ValueError):
        x = [1, 2]
        y = [3, 4]
        fit_theil_sen(x, y)

    with pytest.raises(ValueError):
        x = [[1, 2, 3, 4, 5, 6, 7, 8]]
        y = [3, 4]
        fit_theil_sen(x, y)

    with pytest.raises(ValueError):
        x = [1, 2, 3, 4, 5, 6, 7.7, 8]
        y = [3, 4]
        fit_theil_sen(x, y)

    with pytest.raises(ValueError):
        x = [1, 2, 3, 4, 5, 6, 7.7, 8]
        y = [[1, 2, 3, 4, 5, 6, 7.7],
             [1, 2, 3, 4, 5, 6, 7.7]]
        fit_theil_sen(x, y)

    with pytest.raises(ValueError):
        x = [[1, 2, 3, 4, 5, 6, 7.7, 8]]
        y = [[[1, 2, 3, 4, 5, 6, 7.7, 8]]]
        fit_theil_sen(x, y)


@pytest.mark.parametrize("n,cols", [(20, 40), (21, 40)])
def test_theil_sen_values2d(n, cols):

    x = numpy.arange(n)
    slopes = numpy.linspace(-500, 500, cols)
    intercepts = numpy.linspace(-20, 20, cols)
    y = slopes * x[:, None] + intercepts

    cin, cslope = fit_theil_sen(x, y)

    assert numpy.allclose(slopes, cslope)
    assert numpy.allclose(intercepts, cin)


@pytest.mark.parametrize("n", [20, 21])
def test_theil_sen_values1d(n):

    x = numpy.arange(n)
    slope = 2.0
    intercept = 17
    y = slope * x + intercept

    cin, cslope = fit_theil_sen(x, y)

    assert numpy.allclose(slope, cslope)
    assert numpy.allclose(intercept, cin)