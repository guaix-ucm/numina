import numpy
from ..stats import robust_std
from ..stats import summary


def test_robust_std():
    xdata = numpy.arange(20)

    sigmag = robust_std(xdata)

    assert numpy.allclose([sigmag], [7.0423499999999999])


def test_summary():
    xdata = numpy.arange(50)

    result = summary(xdata)
    expected_result = (0, 12.25, 24.5, 24.5, 36.75, 49, 14.430869689661812,
                       18.161849999999998, 7.7741097000000003,
                       41.225890299999996)

    assert numpy.allclose(result, expected_result)
