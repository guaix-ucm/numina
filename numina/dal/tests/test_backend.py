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

from numina.tests.drptest import create_drp_test
from ..backend import Backend
from numina.exceptions import NoResultFound
import numina.types.qc as qc
import numina.instrument.assembly as asb
from numina.util.context import working_directory


def create_simple_frame():
    from numina.types.frame import DataFrame
    import astropy.io.fits as fits

    simple_img = fits.HDUList([fits.PrimaryHDU()])
    simple_frame = DataFrame(frame=simple_img)
    return simple_frame


def create_simple_structured():
    from numina.types.structured import BaseStructuredCalibration

    obj = BaseStructuredCalibration()
    obj.quality_control = qc.QC.BAD
    return obj


def repeat_my(result_content, result_dir, result_name, storage):
    import numina.store

    with working_directory(result_dir):
        saveres = dict(
            qc=result_content['qc'],
            uuid=result_content['uuid'],
            values={}
        )
        saveres_v = saveres['values']

        result_values = result_content['values']
        for key, obj in result_values.items():
            obj = result_values[key]
            storage.destination = key
            saveres_v[key] = numina.store.dump(obj, obj, storage)

        with open(result_name, 'w') as fd:
            json.dump(saveres, fd)


class Storage(object):
    destination = ''


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

    results_table = {
        1: {'id': 1,
            'instrument': name,
            'mode': "image",
            'oblock_id': 5,
            'task_id': 1,
            'time_create': '2018-07-24T19:12:01',
            'result_dir': 'dum1',
            'result_file': 'dum1/result.json',
            'qc': 'GOOD',
        },
        2: {'id': 2,
            'instrument': name,
            'mode': "sky",
            'oblock_id': 4,
            'task_id': 2,
            'qc': 'BAD',
            'time_create': '2018-07-24T19:12:09',
            'result_file': 'dum2/result.json',
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
            'result_file': 'dum3/result.json',
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

    storage = Storage()
    result_name = 'result.json'

    result1_dir = tmpdir.mkdir('dum1')
    result1_values = dict(
        calib=create_simple_structured(),
        reduced_rss=create_simple_frame(),
        reduced_image=create_simple_frame()
    )
    result1_content = dict(
        qc='GOOD',
        values=result1_values,
        uuid='10000000-10000000-10000000-10000000'
    )
    repeat_my(result1_content, str(result1_dir), result_name, storage)

    result2_dir = tmpdir.mkdir('dum2')

    result2_values = dict(calib=create_simple_structured())
    result2_content = dict(
        qc='BAD',
        values=result2_values,
        uuid='20000000-20000000-20000000-20000000'
    )
    repeat_my(result2_content, str(result2_dir), result_name, storage)

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
    repeat_my(result3_content, str(result3_dir), result_name, storage)

    base = Backend(drps, gentable, components=store, basedir=str(tmpdir))
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
    print(res.content.filename)
    print(backend.basedir)
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
