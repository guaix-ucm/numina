#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

from numina.exceptions import NoResultFound
from numina.tests.drptest import create_drp_test
import numina.core

from ..dictdal import BaseDictDAL
from ..stored import ObservingBlock, StoredProduct


@pytest.fixture
def basedictdal():

    test_drps = create_drp_test(['drptest1.yaml'])
    # com_store = asbl.load_panoply_store(test_drps)
    ob_table = {
        2: dict(id=2, instrument='TEST1', mode="mode1", images=[], children=[], parent=None, facts=None),
        3: dict(id=2, instrument='TEST1', mode="mode2", images=[], children=[], parent=None, facts=None),
    }

    prod_table = {
        'TEST1': [
            {'id': 1, 'type': 'DemoType1', 'tags': {}, 'content': {'demo1': 1}, 'ob': 2},
            {'id': 2, 'type': 'DemoType2', 'tags': {'field2': 'A'}, 'content': {'demo2': 2}, 'ob': 14},
            {'id': 3, 'type': 'DemoType2', 'tags': {'field2': 'B'}, 'content': {'demo2': 3}, 'ob': 15}
        ]
    }

    base = BaseDictDAL(test_drps, ob_table, prod_table, req_table={})
    return base


def test_search_instrument_configuration(basedictdal):
    import numina.instrument.generic

    res = basedictdal.search_instrument_configuration('TEST1', '2017-10-01 12:00:00')

    assert isinstance(res, numina.instrument.generic.InstrumentGeneric)

    res = basedictdal.search_instrument_configuration('TEST1', '2017-10-01 12:00:00')

    assert isinstance(res, numina.instrument.generic.InstrumentGeneric)

    with pytest.raises(ValueError):
        basedictdal.search_instrument_configuration('TEST1', '2011-10-01 12:00:00')

    with pytest.raises(KeyError):
        basedictdal.search_instrument_configuration('TEST2', '2011-10-01 12:00:00')


def test_search_instrument_configuration_from_ob(basedictdal):
    import numina.instrument.generic

    ob = numina.core.ObservationResult(mode=None)

    with pytest.raises(KeyError):
        basedictdal.search_instrument_configuration_from_ob(ob)

    ob = numina.core.ObservationResult(mode='TEST1')
    ob.instrument = 'TEST1'
    res = basedictdal.search_instrument_configuration_from_ob(ob)

    assert isinstance(res, numina.instrument.generic.InstrumentGeneric)

    ob = numina.core.ObservationResult(mode='TEST1')
    ob.instrument = 'TEST1'
    ob.configuration = 'missing'

    with pytest.raises(ValueError):
        basedictdal.search_instrument_configuration_from_ob(ob)

    ob = numina.core.ObservationResult(mode='TEST1')
    ob.instrument = 'TEST1'
    ob.configuration = 'missing'

    with pytest.raises(ValueError):
        basedictdal.search_instrument_configuration_from_ob(ob)


def test_search_instrument_configuration_from_ob2(basedictdal):
    import numina.instrument.generic

    ob = numina.core.ObservationResult(instrument='TEST1', mode='TEST1')
    ob.configuration = '225fcaf2-7f6f-49cc-972a-70fd0aee8e96'

    insconf = basedictdal.search_instrument_configuration_from_ob(ob)
    assert str(insconf.origin.uuid) == '225fcaf2-7f6f-49cc-972a-70fd0aee8e96'


def test_search_instrument_configuration_from_ob3(basedictdal):
    import numina.instrument.generic

    ob = numina.core.ObservationResult(instrument='TEST1', mode='TEST1')
    ob.configuration = '225fcaf2-7f6f-49cc-972a-70fd0aee8e96'

    insconf = basedictdal.search_instrument_configuration_from_ob(ob)
    assert str(insconf.origin.uuid) == '225fcaf2-7f6f-49cc-972a-70fd0aee8e96'



def test_search_oblock(basedictdal):

    with pytest.raises(NoResultFound):
        basedictdal.search_oblock_from_id(obsid=1)

    res = basedictdal.search_oblock_from_id(obsid=2)

    assert isinstance(res, ObservingBlock)

    assert res.id == 2
    assert res.instrument == 'TEST1'


def test_search_recipe(basedictdal):
    from numina.core.utils import AlwaysFailRecipe

    with pytest.raises(KeyError):
        basedictdal.search_recipe('FAIL', 'mode1', 'default')

    with pytest.raises(KeyError):
        basedictdal.search_recipe('TEST1', 'mode1', 'default')

    with pytest.raises(KeyError):
        basedictdal.search_recipe('TEST1', 'fail', 'invalid')

    res = basedictdal.search_recipe('TEST1', 'fail', 'default')
    assert isinstance(res, AlwaysFailRecipe)


def test_search_prod_obsid(basedictdal):

    with pytest.raises(NoResultFound):
        basedictdal.search_prod_obsid('FAIL', 1, 'default')

    with pytest.raises(NoResultFound):
        basedictdal.search_prod_obsid('TEST1', 1, 'default')

    res = basedictdal.search_prod_obsid('TEST1', 2, 'default')
    assert isinstance(res, StoredProduct)


def test_search_prod_req_tags1(basedictdal):
    import numina.core

    class DemoType1(object):
        def name(self):
            return "DemoType1"


    req = numina.core.Requirement(DemoType1, description='Demo1 Requirement')
    ins = 'TEST1'
    tags = {}
    pipeline = 'default'
    res = basedictdal.search_prod_req_tags(req, ins, tags, pipeline)
    assert isinstance(res, StoredProduct)
    assert res.id == 1
    assert res.content == {'demo1': 1}

    assert res.tags == {}


def test_search_prod_req_tags2(basedictdal):
    import numina.core

    class DemoType2(object):
        def name(self):
            return "DemoType2"


    req = numina.core.Requirement(DemoType2, description='Demo2 Requirement')
    ins = 'TEST1'
    tags = {'field2': 'A'}
    pipeline = 'default'
    res = basedictdal.search_prod_req_tags(req, ins, tags, pipeline)
    assert isinstance(res, StoredProduct)
    assert res.id == 2
    assert res.content == {'demo2': 2}

    assert res.tags == {'field2': 'A'}


def test_search_prod_req_tags3(basedictdal):
    import numina.core

    class DemoType2(object):
        def name(self):
            return "DemoType2"


    req = numina.core.Requirement(DemoType2, description='Demo2 Requirement')
    ins = 'TEST1'
    tags = {'field2': 'C'}
    pipeline = 'default'
    with pytest.raises(NoResultFound):
        basedictdal.search_prod_req_tags(req, ins, tags, pipeline)


def test_search_prod_req_tags4(basedictdal):
    import numina.core

    class DemoType2(object):
        def name(self):
            return "DemoType2"


    req = numina.core.Requirement(DemoType2, description='Demo2 Requirement')
    ins = 'TEST1'
    tags = {}
    pipeline = 'default'
    res = basedictdal.search_prod_req_tags(req, ins, tags, pipeline)
    assert isinstance(res, StoredProduct)
    assert res.id == 2
    assert res.content == {'demo2': 2}

    assert res.tags == {'field2': 'A'}
