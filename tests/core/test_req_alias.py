#
# Copyright 2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Tests for alias in RecipeInput"""

import pytest

from numina.core.dataholders import Parameter
from numina.core.recipeinout import RecipeInput


def create_input_class():

    class BB(RecipeInput):
        param1 = Parameter(1, "something1", alias="param3")
        param2 = Parameter(2, "something2")

    return BB


def test_ins_desc_access():

    BB = create_input_class()

    bb = BB(param1=10)

    assert bb.param1 == 10
    assert bb.param3 == 10


def test_ins_desc_access2():

    BB = create_input_class()

    bb = BB(param3=80)

    assert bb.param1 == 80
    assert bb.param3 == 80


def test_ins_attr_access():

    BB = create_input_class()

    bb = BB(param1=80)

    values = {"param2": 2, "param1": 80}

    for key, val in bb.attrs().items():
        assert val == values[key]


def test_setter1():

    BB = create_input_class()

    bb = BB()
    bb.param1 = 80

    values = {"param1": 80, "param2": 2, "param3": 80}

    for key, val in values.items():
        assert val == getattr(bb, key)


def test_setter2():

    BB = create_input_class()

    bb = BB()
    bb.param3 = 80

    values = {"param1": 80, "param2": 2, "param3": 80}

    for key, val in bb.attrs().items():
        assert val == values[key]


def test_alias_unknown_attribute():

    BB = create_input_class()
    bb = BB()

    with pytest.raises(AttributeError):
        bb.param4


def test_alias_field_has_priority():

    class BB(RecipeInput):
        param1 = Parameter(1, "something1", alias="param2")
        param2 = Parameter(2, "something2")

    bb = BB()
    assert bb.param1 == 1
    assert bb.param2 == 2


def test_alias_in_recipe():
    from numina.core.recipes import BaseRecipe

    class RecipeBase1(BaseRecipe):
        param1 = Parameter(1, "something1", alias="param3")

    class RecipeTest(RecipeBase1):
        param2 = Parameter(2, "something2")

    rinput = RecipeTest.create_input(param3=80)
    assert rinput.param1 == 80
    assert rinput.param3 == 80
