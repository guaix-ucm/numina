import numpy

from numina.array.peaks.detrend import detrend

numpy.random.seed(12812818)

def test_compute_trend():
    numpy.random.seed(12812818)

    xx = numpy.arange(100)
    yy = 700 + 3.2 * (xx / 100.0)
    yy = numpy.random.normal(loc=yy, scale=2.0)

    trend = detrend(yy)

    maxt = trend.max()
    mint = trend.min()
    expected = [704.812599, 698.595538331]

    numpy.testing.assert_array_almost_equal([maxt, mint], expected)
