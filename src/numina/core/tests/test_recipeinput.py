#
# Copyright 2015-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Tests for RecipeInput and RecipeResult"""

from ..dataholders import Parameter
from ..recipeinout import RecipeInput


def create_input_class():

    class BB(RecipeInput):
        param1 = Parameter(1, 'something1')
        param2 = Parameter(2, 'something2')

        def mayfun(self):
            pass

    return BB


def test_class_construction():

    BB = create_input_class()

    assert issubclass(BB, RecipeInput)


def test_class_desc_access():

    BB = create_input_class()

    assert BB.param1 is getattr(BB, 'param1')
    assert BB.param2 is getattr(BB, 'param2')

    assert isinstance(BB.param1, Parameter)
    assert isinstance(BB.param2, Parameter)


def test_class_desc_set():

    BB = create_input_class()

    BB.param3 = Parameter(3, 'something3')

    assert BB.param3 is getattr(BB, 'param3')
    assert isinstance(BB.param3, Parameter)
    assert BB.param3 is BB.stored()['param3']
    assert BB.param3.dest == 'param3'


def test_class_destination_set():

    class BB(RecipeInput):
        param3h2hd = Parameter(1, 'something1', destination="param3")

    assert BB.param3 is getattr(BB, 'param3')
    assert isinstance(BB.param3, Parameter)
    assert BB.param3 is BB.stored()['param3']
    assert BB.param3.dest == 'param3'


def test_class_desc_stored():

    BB = create_input_class()

    stored = BB.stored()

    assert BB.param1 is stored['param1']
    assert BB.param2 is stored['param2']


def test_ins_desc_access():

    BB = create_input_class()

    bb = BB(param1=80)

    values = {'param2': 2, 'param1': 80}

    for key, val in values.items():
        ival = getattr(bb, key)
        assert val == ival

    # These values are not stored in __dict__
    for key in values:
        assert key not in bb.__dict__


def test_ins_attr_access():

    BB = create_input_class()

    bb = BB(param1=80)

    values = {'param2': 2, 'param1': 80}

    for key, val in bb.attrs().items():
        assert val == values[key]

    # These values are not stored in __dict__
    for key in values:
        assert key not in bb.__dict__


def test_class_nondesc_access():

    BB = create_input_class()
    bb = BB(param1=80)

    values = {'param2': 2, 'param1': 80}

    # If we insert a new attribute,it goes to __dict__
    bb.otherattr = 100

    for key, val in bb.attrs().items():
        assert val == values[key]

    assert 'otherattr' in bb.__dict__