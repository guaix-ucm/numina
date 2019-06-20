#
# Copyright 2018-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

from numina.tests.drptest import create_drp_test
import numina.instrument.assembly as asb

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
    # Load instrument profiles
    pkg_paths = ['numina.drps.tests.configs']
    store = asb.load_paths_store(pkg_paths)
    base = HybridDAL(drps, ob_table, gentable, {}, components=store)

    return base


def test_skip_reserved(hybriddal):

    ss_ids = list(hybriddal.search_session_ids())

    assert ss_ids == [2, 3, 4, 5, 30, 40]


def test_parent_inserted(hybriddal):

    obsres = hybriddal.search_oblock_from_id(2)
    print('OBSR', obsres.__dict__)
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
