
import numpy
import numpy.random
import pytest

from ..background import background_estimator
from ..background import create_background_map


def test_back1():
    bdata = 300 * numpy.ones((50, 50))
    back, std = background_estimator(bdata)
    assert numpy.allclose([back, std], [300, 0])


def test_back2():
    res = (301.23398492452793, 12.277994974842644)
    bdata = 300 * numpy.ones((50, 50))
    bdata[20:25, 20:25] = 700
    r = background_estimator(bdata)
    assert numpy.allclose(r, res)



def test_back_crowded():
    res = (318.5390526053346, 73.37968420177819)
    bdata = 300 * numpy.ones((50, 50))
    bdata[20:25, 20:25] = 700
    bdata[30:35, 30:35] = 700
    bdata[10:15, 10:15] = 700
    bdata[10:15, 20:25] = 700
    bdata[10:15, 30:35] = 700
    bdata[20:25, 30:35] = 700
    r = background_estimator(bdata)
    assert numpy.allclose(r, res)


@pytest.mark.xfail
def test_background_map():
    numpy.random.seed(seed=938483)
    bck = numpy.random.normal(1100, 32, (512,512))
    nd, ns = create_background_map(bck, 8, 8)
    res = (1100.055068749893, 31.478283226675636)
    assert numpy.allclose([nd.mean(), ns.mean()], res)
