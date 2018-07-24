#
# Copyright 2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

from numina.tests.drptest import create_drp_test
from ..dictdal import Backend

@pytest.fixture
def backend():

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
        dict(id=400, instrument=name, mode="raiz", images=[], children=[30 , 40]),
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
    #gentable['oblocks'] = ob_table

    base = Backend(drps, gentable)
    base.add_obs(ob_table)

    return base


def test_skip_reserved(backend):

    ss_ids = list(backend.search_session_ids())

    assert ss_ids == [2, 3, 4, 5, 30, 40]


def test_parent_inserted(backend):

    obsres = backend.search_oblock_from_id(2)
    assert obsres.parent == 30

    obsres = backend.search_oblock_from_id(4)
    assert obsres.parent == 40

    obsres = backend.search_oblock_from_id(30)
    assert obsres.parent == 400

    obsres = backend.search_oblock_from_id(400)
    assert obsres.parent is None


def test_previous_obsid(backend):

    obsres = backend.search_oblock_from_id(5)
    previd = backend.search_previous_obsres(obsres, node='prev')
    assert list(previd) == [4, 3, 2, 1]

    obsres = backend.search_oblock_from_id(5)
    previd = backend.search_previous_obsres(obsres, node='prev-rel')
    assert list(previd) == [4]

    obsres = backend.search_oblock_from_id(4)
    previd = backend.search_previous_obsres(obsres, node='prev-rel')
    assert list(previd) == []