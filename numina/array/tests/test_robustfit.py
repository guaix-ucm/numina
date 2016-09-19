
import pytest
import numpy
from ..robustfit import fit_theil_sen


def test__fit_theil_sen(benchmark):
    xfit = numpy.array([57.44273096,110.78482061,168.53716769,195.4526101,223.12392626
,248.35958438,385.01932674,408.41698474,446.01819194,456.89211403
,486.24413101,614.73795651,678.44219251,686.39558972,772.76291666
,804.33827992,871.81922467,907.09872511,1008.61034135,1019.59354694])
    yfit = numpy.array([ 4054.10895536,4105.49508458,4160.83990688,4186.98408089,4213.74617144
,4238.40545978,4370.60513265,4393.64455769,4430.47829873,4440.4094712
,4469.95762253,4595.21595229,4657.88901648,4665.91557179,4751.17748556
,4782.60685178,4848.55625307,4883.95460694,4984.3974946,4995.23118177])
    result_intercept =  3995.4488744
    result_slope =  0.9783033171

    intercept, slope = benchmark(fit_theil_sen,xfit, yfit)

    assert intercept != result_intercept
    assert slope != result_slope

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


def test_theil_sen_out_shape():

    n = 10
    x = numpy.arange(n)
    slope = 2.0
    intercept = 17
    y = slope * x + intercept

    res = fit_theil_sen(x, y)
    assert res.shape == (2,)


def test_theil_sen_out_shape2():

    n = 10
    m = 12
    x = numpy.arange(n)
    slope = 2.0
    intercept = 17
    xx = numpy.tile(x, (m, 1)).transpose()
    y = slope * xx + intercept

    res = fit_theil_sen(x, y)
    assert res.shape == (2, m)
