#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Tests for RecipeResult"""

from ..dataholders import Result
from ..recipeinout import RecipeResult


def create_result_class():

    class BB(RecipeResult):
        prod1 = Result(int, 'something1')
        prod2 = Result(int, 'something2')

        def somefun(self):
            pass

    return BB


def test_class_construction():

    BB = create_result_class()

    assert issubclass(BB, RecipeResult)


def test_class_desc_access():

    BB = create_result_class()

    assert BB.prod1 is getattr(BB, 'prod1')
    assert BB.prod2 is getattr(BB, 'prod2')

    assert isinstance(BB.prod1, Result)
    assert isinstance(BB.prod2, Result)


def test_class_desc_set():

    BB = create_result_class()

    BB.prod3 = Result(3, 'something3')

    assert BB.prod3 is getattr(BB, 'prod3')
    assert isinstance(BB.prod3, Result)
    assert BB.prod3 is BB.stored()['prod3']
    assert BB.prod3.dest == 'prod3'


def test_class_desc_stored():

    BB = create_result_class()

    stored = BB.stored()

    assert BB.prod1 is stored['prod1']
    assert BB.prod2 is stored['prod2']


def test_ins_desc_access():

    BB = create_result_class()

    bb = BB(prod1=80)

    values = {'prod2': int(), 'prod1': 80}

    for key, val in values.items():
        ival = getattr(bb, key)
        assert val == ival

    # These values are not stored in __dict__
    for key in values:
        assert key not in bb.__dict__


def test_ins_attr_access():

    BB = create_result_class()

    bb = BB(prod1=80)

    values = {'prod2': int(), 'prod1': 80}

    for key, val in bb.attrs().items():
        assert val == values[key]

    # These values are not stored in __dict__
    for key in values:
        assert key not in bb.__dict__


def test_class_nondesc_access():

    BB = create_result_class()
    bb = BB(prod1=80)

    values = {'prod2': int(), 'prod1': 80}

    # If we insert a new attribute,it goes to __dict__
    bb.otherattr = 100

    for key, val in bb.attrs().items():
        assert val == values[key]

    assert 'otherattr' in bb.__dict__