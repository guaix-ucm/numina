#
# Copyright 2018 Universidad Complutense de Madrid
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

import pytest

from numina.tests.drptest import create_drp_test
from ..dictdal import HybridDAL

@pytest.fixture
def hybriddal():

    drps = create_drp_test(['drpclodia.yaml'])
    name = 'CLODIA'
    ob_table = [
        dict(id=1, instrument=name, mode="sky", images=[], children=[],
             enabled=False
             ),
        dict(id=2, instrument=name, mode="sky", images=[], children=[]),
        dict(id=3, instrument=name, mode="image", images=[], children=[]),
        dict(id=4, instrument=name, mode="sky", images=[], children=[]),
        dict(id=5, instrument=name, mode="image", images=[], children=[]),
        dict(id=30, instrument=name, mode="mosaic", images=[], children=[2, 3]),
        dict(id=40, instrument=name, mode="mosaic", images=[], children=[4, 5]),
        dict(id=400, instrument=name, mode="raiz", images=[], children=[30 ,40]),
    ]

    prod_table = {
        'TEST1': [
            {'id': 1, 'type': 'DemoType1', 'tags': {},
             'content': {'demo1': 1}, 'ob': 2},
            {'id': 2, 'type': 'DemoType2', 'tags': {'field2': 'A'},
             'content': {'demo2': 2}, 'ob': 14},
            {'id': 3, 'type': 'DemoType2', 'tags': {'field2': 'B'},
             'content': {'demo2': 3}, 'ob': 15}
        ]
    }

    gentable = {}
    gentable['products'] = prod_table
    gentable['requirements'] = {}

    base = HybridDAL(drps, ob_table, gentable, {})

    return base


def test_skip_reserved(hybriddal):

    ss_ids = list(hybriddal.search_session_ids())

    assert ss_ids == [2, 3, 4, 5, 30, 40]


def test_parent_inserted(hybriddal):

    obsres = hybriddal.search_oblock_from_id(2)
    assert obsres.parent == 30

    obsres = hybriddal.search_oblock_from_id(4)
    assert obsres.parent == 40

    obsres = hybriddal.search_oblock_from_id(30)
    assert obsres.parent == 400

    obsres = hybriddal.search_oblock_from_id(400)
    assert obsres.parent is None


def test_previous_obsid(hybriddal):

    obsres = hybriddal.search_oblock_from_id(5)
    previd = hybriddal.search_previous_obsres(obsres, node='prev')
    assert list(previd) == [4, 3, 2, 1]

    obsres = hybriddal.search_oblock_from_id(5)
    previd = hybriddal.search_previous_obsres(obsres, node='prev-rel')
    assert list(previd) == [4]

    obsres = hybriddal.search_oblock_from_id(4)
    previd = hybriddal.search_previous_obsres(obsres, node='prev-rel')
    assert list(previd) == []
