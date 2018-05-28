#
# Copyright 2017-2018 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Unit test for MultiType adapter"""

import pytest

import numina.exceptions
from ..datatype import DataType
from ..multitype import MultiType


class AVal(object):
    pass


class BVal(object):
    pass


class TypeA(DataType):

    def __init__(self):
        super(TypeA, self).__init__(ptype=AVal)

    def convert(self, obj):
        return TypeA()

    def tag_names(self):
        return ['domeA', 'domeA2']


class TypeB(DataType):

    def __init__(self):
        super(TypeB, self).__init__(ptype=BVal)

    def convert(self, obj):
        return TypeB()

    def tag_names(self):
        return ['domeB']


def test_convert_in():

    multi = MultiType(TypeA, TypeB)

    with pytest.raises(ValueError):
        multi.convert_in(None)


def test_validate1():
    multi = MultiType(TypeA, TypeB)

    # testing without selection the internal type
    # It should validate both types
    assert multi.validate(AVal())
    assert multi.validate(BVal())

    # And fail with anything else
    with pytest.raises(numina.exceptions.ValidationError) as excinfo:
        multi.validate(3)

    intl_exception = excinfo.value
    intl_args = intl_exception.args[1]
    assert intl_args == (AVal, BVal)


def test_validate2():

    multi = MultiType(TypeA, TypeB)

    # Simulate valid query
    multi._current = multi.type_options[0]

    # testing with selection of the internal type
    # It should validate TypeA only
    assert multi.validate(AVal())

    with pytest.raises(numina.exceptions.ValidationError):
        multi.validate(BVal())

    with pytest.raises(numina.exceptions.ValidationError):
        multi.validate(3)


def test_tag_names1():

    multi = MultiType(TypeA, TypeB)

    assert multi.tag_names() == ['domeA', 'domeA2', 'domeB']



def test_tag_names2():

    multi = MultiType(TypeA, TypeB)
    multi._current = multi.type_options[1]

    assert multi.tag_names() == ['domeB']
