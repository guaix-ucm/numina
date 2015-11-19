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


"""Unit test for taggers"""

import pytest

import astropy.io.fits as fits

from ..dataframe import DataFrame
from ..oresult import ObservationResult
from ..taggers import get_tags_from_full_ob


def test_empty_ob():

    ob = ObservationResult()
    tags = get_tags_from_full_ob(ob)

    assert len(tags) == 0


def test_init_ob():

    img1 = fits.PrimaryHDU(data=[1,2,3])
    frame1 = DataFrame(frame=fits.HDUList(img1))

    ob = ObservationResult()
    ob.frames = [frame1]
    tags = get_tags_from_full_ob(ob)

    assert len(tags) == 0


def test_header_key1_ob():

    img1 = fits.PrimaryHDU(data=[1,2,3], header=fits.Header())
    img1.header['FILTER'] = 'FILTER-A'
    img1.header['READM'] = 'MOD1'
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1,2,3], header=fits.Header())
    img2.header['FILTER'] = 'FILTER-A'
    img1.header['READM'] = 'MOD2'
    frame2 = DataFrame(frame=fits.HDUList(img1))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]
    tags = get_tags_from_full_ob(ob, reqtags=['FILTER'])

    assert tags == {'FILTER': 'FILTER-A'}


def test_header_key1_mis():

    img1 = fits.PrimaryHDU(data=[1,2,3], header=fits.Header())
    img1.header['FILTER'] = 'FILTER-A'
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1,2,3], header=fits.Header())
    img2.header['FILTER'] = 'FILTER-B'
    frame2 = DataFrame(frame=fits.HDUList(img2))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]

    with pytest.raises(ValueError):
        get_tags_from_full_ob(ob, reqtags=['FILTER'])


def test_header_key2_ob():

    img1 = fits.PrimaryHDU(data=[1,2,3], header=fits.Header())
    img1.header['FILTER'] = 'FILTER-A'
    img1.header['READM'] = 'MOD1'
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1,2,3], header=fits.Header())
    img2.header['FILTER'] = 'FILTER-A'
    img1.header['READM'] = 'MOD1'
    frame2 = DataFrame(frame=fits.HDUList(img1))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]
    tags = get_tags_from_full_ob(ob, reqtags=['FILTER', 'READM'])

    assert tags == {'FILTER': 'FILTER-A', 'READM': 'MOD1'}