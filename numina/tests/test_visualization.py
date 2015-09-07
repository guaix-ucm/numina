#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

'''Unit test for visualization module'''

from  ..visualization import ZScaleInterval

import pytest

import numpy.random
from numpy.testing import assert_allclose


@pytest.fixture(scope='module')
def narray():
    numpy.random.seed(2339382849)
    data = 300 + 10 * numpy.random.normal(size=(1024, 1024))
    return data


def test_zscale_interval(narray):

    zscale = ZScaleInterval()
    z12 = zscale.get_limits(narray)
    res = (250.43783329355836, 347.76843337965028)
    assert_allclose(z12, res)

def test_zscale_interval_contrast_030(narray):

    zscale = ZScaleInterval(contrast=0.30)
    z12 = zscale.get_limits(narray)
    res = (257.46025418257619, 342.5343627342458)
    assert_allclose(z12, res)

def test_zscale_interval_contrast_0(narray):

    zscale = ZScaleInterval(contrast=0)
    z12 = zscale.get_limits(narray)
    res = (287.23619217566056, 312.75842474116143)
    assert_allclose(z12, res)

def test_zscale_interval_contrast_minmax():

    zscale = ZScaleInterval()
    z12 = zscale.get_limits([299, 301, 302, 303])
    res = (299, 303)
    assert_allclose(z12, res)
