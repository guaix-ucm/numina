import numpy
from ..stats import robust_std
from ..stats import summary


def test_robust_std():
    xdata = numpy.arange(20)

    sigmag = robust_std(xdata)

    assert numpy.allclose([sigmag], [7.0423499999999999])


def test_robust_std_2d():
    xdata = numpy.arange(20).reshape((5,4))

    sigmag = robust_std(xdata)

    assert numpy.allclose([sigmag], [7.0423499999999999])


def test_summary():
    xdata = numpy.arange(50)

    result = summary(xdata)

    expected_result = {
        'npoints': 50,
        'minimum': 0,
        'percentile25': 12.25,
        'median': 24.5,
        'mean': 24.5,
        'percentile75': 36.75,
        'maximum': 49,
        'std': 14.430869689661812,
        'robust_std': 18.161849999999998,
        'percentile15': 7.7741097000000003,
        'percentile84': 41.225890299999996
    }

    for key, val in result.items():
        assert numpy.allclose(result[key], expected_result[key])
