#
# Copyright 2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Tests for alias in RecipeInput"""

from ..dataholders import Parameter
from ..recipeinout import RecipeInput


def create_input_class():

    class BB(RecipeInput):
        param1 = Parameter(1, 'something1', alias='param3')
        param2 = Parameter(2, 'something2')

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

    values = {'param2': 2, 'param1': 80}

    for key, val in bb.attrs().items():
        assert val == values[key]


def test_setter1():

    BB = create_input_class()

    bb = BB()
    bb.param1 = 80

    values = {'param1': 80, 'param2': 2, 'param3': 80}

    for key, val in values.items():
        assert val == getattr(bb, key)


def test_setter2():

    BB = create_input_class()

    bb = BB()
    bb.param3 = 80

    values = {'param1': 80, 'param2': 2, 'param3': 80}

    for key, val in bb.attrs().items():
        assert val == values[key]
