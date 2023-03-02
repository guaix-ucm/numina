#
# Copyright 2018-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import os.path
import json
import uuid

import pytest

from numina.util.jsonencoder import ExtEncoder
from numina.tests.drptest import create_drp_test
from ..backend import Backend
from numina.exceptions import NoResultFound
import numina.types.qc as qc
import numina.instrument.assembly as asb
import numina.core.oresult
from numina.util.context import working_directory
import numina.tests.simpleobj as simpleobj


def repeat_my(result_content, result_name):
    import numina.store

    saveres = dict(
        qc=result_content['qc'],
        uuid=result_content['uuid'],
        values={}
    )
    saveres_v = saveres['values']

    result_values = result_content['values']
    for key, obj in result_values.items():
        obj = result_values[key]
        saveres_v[key] = numina.store.dump(obj, obj, key)

    with open(result_name, 'w') as fd:
        json.dump(saveres, fd, indent=2, cls=ExtEncoder)


@pytest.fixture
def backend(tmpdir):

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
    result_name = 'result.json'
    results_table = {
        1: {'id': 1,
            'instrument': name,
            'mode': "image",
            'oblock_id': 5,
            'task_id': 1,
            'time_create': '2018-07-24T19:12:01',
            'result_dir': 'dum1',
            'result_file': result_name,
            'qc': 'GOOD',
        },
        2: {'id': 2,
            'instrument': name,
            'mode': "sky",
            'oblock_id': 4,
            'task_id': 2,
            'qc': 'BAD',
            'time_create': '2018-07-24T19:12:09',
            'result_file': result_name,
            'result_dir': 'dum2',
        },
        3: {'id': 3,
            'instrument': name,
            'mode': "image",
            'oblock_id': 5,
            'task_id': 3,
            'time_create': '2018-07-24T19:12:11',
            'qc': 'GOOD',
            'result_dir': 'dum3',
            'result_file': result_name,
        },
    }

    gentable = {}
    gentable['products'] = prod_table
    gentable['requirements'] = {}
    gentable['results'] = results_table
    #gentable['oblocks'] = ob_table
    # Load instrument profiles
    pkg_paths = ['numina.drps.tests.configs']
    store = asb.load_paths_store(pkg_paths)

    result_name = 'result.json'

    result1_dir = tmpdir.mkdir('dum1')
    result1_values = dict(
        calib=simpleobj.create_simple_structured(),
        reduced_rss=simpleobj.create_simple_frame(),
        reduced_image=simpleobj.create_simple_frame()
    )
    result1_content = dict(
        qc='GOOD',
        values=result1_values,
        uuid='10000000-10000000-10000000-10000000'
    )
    with working_directory(str(result1_dir)):
        repeat_my(result1_content, result_name)

    result2_dir = tmpdir.mkdir('dum2')

    result2_values = dict(calib=simpleobj.create_simple_structured())
    result2_content = dict(
        qc='BAD',
        values=result2_values,
        uuid='20000000-20000000-20000000-20000000'
    )
    with working_directory(str(result2_dir)):
        repeat_my(result2_content, result_name)

    result3_dir = tmpdir.mkdir('dum3')

    result3_values = dict(
        reduced_rss='reduced_rss.fits',
        reduced_image='reduced_image.fits'
    )
    result3_content = dict(
        qc='BAD',
        values=result3_values,
        uuid='30000000-30000000-30000000-30000000'
    )
    with working_directory(str(result3_dir)):
        repeat_my(result3_content, result_name)

    base = Backend(drps, gentable, components=store, basedir=str(tmpdir))
    base.add_obs(ob_table)

    return base


@pytest.fixture
def backend_empty(tmpdir):
    drps = create_drp_test(['drpclodia.yaml'])
    pkg_paths = ['numina.drps.tests.configs']
    store = asb.load_paths_store(pkg_paths)
    base = Backend(drps, {}, components=store, basedir=str(tmpdir))
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
    print(type(obsres))
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
    relpath = os.path.relpath(res.content.filename, backend.basedir)
    assert relpath == os.path.join('dum3', 'reduced_rss.fits')


def test_search_result_id_notfound(backend):
    from numina.types.frame import DataFrameType

    tipo = DataFrameType()
    node_id = 2
    field = 'reduced_rss'

    with pytest.raises(NoResultFound):
        backend.search_result_id(node_id, tipo, field, mode=None)


def test_build_recipe_result(backend):
    from numina.types.dataframe import DataFrame
    from numina.types.structured import BaseStructuredCalibration

    obj_uuid = '10000000-10000000-10000000-10000000'
    res = backend.build_recipe_result(result_id=1)

    assert res.qc == qc.QC.GOOD

    assert hasattr(res, 'reduced_rss')
    assert isinstance(res.reduced_rss, DataFrame)

    assert hasattr(res, 'reduced_image')
    assert isinstance(res.reduced_image, DataFrame)

    assert hasattr(res, 'calib')
    assert isinstance(res.calib, BaseStructuredCalibration)

    assert res.uuid == uuid.UUID(obj_uuid)


def test_build_recipe_result2(backend):
    from numina.types.structured import BaseStructuredCalibration
    obj_uuid = '20000000-20000000-20000000-20000000'
    res = backend.build_recipe_result(result_id=2)

    assert res.qc == qc.QC.BAD

    assert hasattr(res, 'calib')
    assert isinstance(res.calib, BaseStructuredCalibration)

    assert res.uuid == uuid.UUID(obj_uuid)


def test_ago(backend_empty):

    ob1 = {'id': 100, 'instrument': 'CLODIA', 'mode': 'image', 'labels': {'obsid_wl': 400}}
    ob2 = {'id': 200, 'instrument': 'CLODIA', 'mode': 'image', 'labels': {'obsid_wl': 400}}
    ob3 = {'id': 300, 'instrument': 'CLODIA', 'mode': 'image'}
    ob4 = {'id': 400, 'instrument': 'CLODIA', 'mode': 'image'}

    backend_empty.add_obs([ob1, ob2, ob3, ob4])

    oblock1 = backend_empty.oblock_from_id(100)
    assert isinstance(oblock1, numina.core.oresult.ObservingBlock)

    assert oblock1.labels == ob1['labels']

    oblock2 = backend_empty.oblock_from_id(200)
    assert isinstance(oblock2, numina.core.oresult.ObservingBlock)

    obsres1 = backend_empty.obsres_from_oblock(oblock1)

    assert isinstance(obsres1, numina.core.oresult.ObservationResult)
