#
# Copyright 2016-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import pytest

from numina.exceptions import NoResultFound
from numina.tests.drptest import create_drp_test
import numina.core

from ..dictdal import BaseDictDAL
from ..stored import StoredProduct


@pytest.fixture
def basedictdal():

    test_drps = create_drp_test(["drptest1.yaml"])
    # com_store = asbl.load_panoply_store(test_drps)
    ob_table = {
        2: dict(id=2, instrument="TEST1", mode="mode1", images=[], children=[], parent=None, facts=None),
        3: dict(id=2, instrument="TEST1", mode="mode2", images=[], children=[], parent=None, facts=None),
    }

    prod_table = {
        "TEST1": {
            "225fcaf2-7f6f-49cc-972a-70fd0aee8e96": [
                {"id": 1, "type": "DemoType1", "tags": {}, "content": {"demo1": 1}, "ob": 2},
                {"id": 2, "type": "DemoType2", "tags": {"field2": "A"}, "content": {"demo2": 2}, "ob": 14},
                {"id": 3, "type": "DemoType2", "tags": {"field2": "B"}, "content": {"demo2": 3}, "ob": 15},
            ]
        }
    }

    base = BaseDictDAL(test_drps, ob_table, prod_table, req_table={})
    return base


def test_search_oblock(basedictdal):

    with pytest.raises(KeyError):
        basedictdal.oblock_from_id(obsid=1)

    res = basedictdal.oblock_from_id(obsid=2)

    assert isinstance(res, numina.core.oresult.ObservingBlock)

    assert res.id == 2
    assert res.instrument == "TEST1"


def test_search_recipe(basedictdal):
    from numina.core.utils import AlwaysFailRecipe

    with pytest.raises(KeyError):
        basedictdal.search_recipe("FAIL", "mode1", "default")

    with pytest.raises(KeyError):
        basedictdal.search_recipe("TEST1", "mode1", "default")

    with pytest.raises(KeyError):
        basedictdal.search_recipe("TEST1", "fail", "invalid")

    res = basedictdal.search_recipe("TEST1", "fail", "default")
    assert isinstance(res, AlwaysFailRecipe)


def test_search_prod_type_tags1(basedictdal):

    class DemoType1:
        def name(self):
            return "DemoType1"

    req = numina.core.Requirement(DemoType1, description="Demo1 Requirement")
    ins = "TEST1"
    version = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
    tags = {}
    pipeline = "default"
    res = basedictdal.search_prod_type_tags(req.type, ins, version, tags, pipeline)
    assert isinstance(res, StoredProduct)
    assert res.id == 1
    assert res.content == {"demo1": 1}

    assert res.tags == {}


def test_search_prod_type_tags2(basedictdal):

    class DemoType2:
        def name(self):
            return "DemoType2"

    req = numina.core.Requirement(DemoType2, description="Demo2 Requirement")
    ins = "TEST1"
    version = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
    tags = {"field2": "A"}
    pipeline = "default"
    res = basedictdal.search_prod_type_tags(req.type, ins, version, tags, pipeline)
    assert isinstance(res, StoredProduct)
    assert res.id == 2
    assert res.content == {"demo2": 2}

    assert res.tags == {"field2": "A"}


def test_search_prod_type_tags3(basedictdal):

    class DemoType2:
        def name(self):
            return "DemoType2"

    req = numina.core.Requirement(DemoType2, description="Demo2 Requirement")
    ins = "TEST1"
    version = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
    tags = {"field2": "C"}
    pipeline = "default"
    with pytest.raises(NoResultFound):
        basedictdal.search_prod_type_tags(req.type, ins, version, tags, pipeline)


def test_search_prod_type_tags4(basedictdal):
    class DemoType2:
        def name(self):
            return "DemoType2"

    req = numina.core.Requirement(DemoType2, description="Demo2 Requirement")
    ins = "TEST1"
    version = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"
    tags = {}
    pipeline = "default"
    res = basedictdal.search_prod_type_tags(req.type, ins, version, tags, pipeline)
    assert isinstance(res, StoredProduct)
    assert res.id == 2
    assert res.content == {"demo2": 2}

    assert res.tags == {"field2": "A"}
