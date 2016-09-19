
import pytest
import numpy
from ..mode import mode_half_sample, mode_sex


def test_mode_halfsample():
    numpy.random.seed(1392838)
    a = numpy.random.normal(1000, 200, size=1000)
    a[:100] = numpy.random.normal(2000, 300, size=100)
    b = numpy.sort(a)
    m = mode_half_sample(b, is_sorted=True)
    assert numpy.allclose(m, 1041.9327885039545)


def test_mode_sextractor():
    numpy.random.seed(1392838)
    a = numpy.random.normal(1000, 200, size=1000)
    a[:100] = numpy.random.normal(2000, 300, size=100)
    b = numpy.sort(a)
    m = mode_sex(b)
    assert numpy.allclose(m, 912.78563284477332)
