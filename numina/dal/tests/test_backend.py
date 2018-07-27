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
from ..backend import Backend
from numina.exceptions import NoResultFound
import numina.types.qc as qc

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

    results_table = {
        1: {'id': 1,
            'instrument': name,
            'mode': "image",
            'oblock_id': 5,
            'task_id': 1,
            'time_create': '2018-07-24T19:12:01',
            'directory': 'dum1',
            'qc': 'GOOD',
            'values': [
                {'content': 'reduced_rss.fits', 'name': 'reduced_rss',
                 'type': 'DataFrameType', 'type_fqn': 'numina.types.frame.DataFrameType'},
                {'content': 'reduced_image.fits', 'name': 'reduced_image', 'type': 'DataFrameType',
                 'type_fqn': 'numina.types.frame.DataFrameType'},
                {'content': 'calib.json', 'name': 'calib', 'type': 'Other',
                 'type_fqn': 'numina.types.frame.Other'},
            ]
        },
        2: {'id': 2,
            'instrument': name,
            'mode': "sky",
            'oblock_id': 4,
            'task_id': 2,
            'qc': 'BAD',
            'time_create': '2018-07-24T19:12:09',
            'directory': 'dum2',
            'result_file': 'result.json',
            'values': [],
        },
        3: {'id': 3,
            'instrument': name,
            'mode': "image",
            'oblock_id': 5,
            'task_id': 3,
            'time_create': '2018-07-24T19:12:11',
            'qc': 'GOOD',
            'directory': 'dum3',
            'values': [
                {'content': 'reduced_rss.fits', 'name': 'reduced_rss',
                 'type': 'DataFrameType', 'type_fqn': 'numina.types.frame.DataFrameType'},
                {'content': 'reduced_image.fits', 'name': 'reduced_image', 'type': 'DataFrameType',
                 'type_fqn': 'numina.types.frame.DataFrameType'},
            ]
        },
    }

    gentable = {}
    gentable['products'] = prod_table
    gentable['requirements'] = {}
    gentable['results'] = results_table
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


def test_search_result_id(backend):
    from numina.types.frame import DataFrameType
    from numina.types.dataframe import DataFrame

    tipo = DataFrameType()
    node_id = 5
    field = 'reduced_rss'
    res = backend.search_result_id(node_id, tipo, field, mode=None)

    assert isinstance(res.content, DataFrame)
    assert res.content.filename == 'dum3/reduced_rss.fits'


def test_search_result_id_notfound(backend):
    from numina.types.frame import DataFrameType

    tipo = DataFrameType()
    node_id = 2
    field = 'reduced_rss'

    with pytest.raises(NoResultFound):
        backend.search_result_id(node_id, tipo, field, mode=None)


def test_build_recipe_result(backend, tmpdir):
    from numina.types.dataframe import DataFrame
    from numina.types.structured import BaseStructuredCalibration, writeto
    from numina.util.context import working_directory

    p = tmpdir.join("calib.json")

    bs = BaseStructuredCalibration()
    writeto(bs, str(p))

    with working_directory(str(tmpdir)):
        res = backend.build_recipe_result(result_id=1)

    assert res.qc == qc.QC.GOOD

    assert hasattr(res, 'reduced_rss')
    assert isinstance(res.reduced_rss, DataFrame)

    assert hasattr(res, 'reduced_image')
    assert isinstance(res.reduced_image, DataFrame)

    assert hasattr(res, 'calib')
    assert isinstance(res.calib, BaseStructuredCalibration)


def test_build_recipe_result2(backend, tmpdir):
    import numina.store
    from numina.types.structured import BaseStructuredCalibration
    from numina.util.context import working_directory
    import json

    resd = {}
    resd['qc'] = 'BAD'
    saveres = {}
    resd['values'] = saveres
    obj = BaseStructuredCalibration()
    obj.quality_control = qc.QC.BAD
    obj_uuid = obj.uuid

    class Storage(object):
        pass

    storage = Storage()
    storage.destination = 'calib'

    with working_directory(str(tmpdir)):
        saveres['calib'] = numina.store.dump(obj, obj, storage)

        p = tmpdir.join("result.json")
        p.write(json.dumps(resd))

    with working_directory(str(tmpdir)):
        res = backend.build_recipe_result2(result_id=2)

    assert res.qc == qc.QC.BAD

    #assert hasattr(res, 'reduced_rss')
    #assert isinstance(res.reduced_rss, DataFrame)

    #assert hasattr(res, 'reduced_image')
    #assert isinstance(res.reduced_image, DataFrame)

    assert hasattr(res, 'calib')
    assert isinstance(res.calib, BaseStructuredCalibration)

    assert res.calib.uuid == obj_uuid
