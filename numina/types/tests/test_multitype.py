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
from ..structured import BaseStructuredCalibration

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


class Structured1(BaseStructuredCalibration):
    def __init__(self):
        super(Structured1, self).__init__()
        self.struct1 = 1

    def __getstate__(self):
        state = super(Structured1, self).__getstate__()
        state['struct1'] = self.struct1
        return state

    def __setstate__(self, state):
        super(Structured1, self).__setstate__(state)
        self.struct1 = state['struct1']


class Structured2(BaseStructuredCalibration):
    def __init__(self):
        super(Structured2, self).__init__()
        self.struct2 = 2

    def __getstate__(self):
        state = super(Structured2, self).__getstate__()
        state['struct2'] = self.struct2
        return state

    def __setstate__(self, state):
        super(Structured2, self).__setstate__(state)
        self.struct2 = state['struct2']


class Structured3(BaseStructuredCalibration):
    def __init__(self):
        super(Structured3, self).__init__()
        self.struct3 = 3

    def __getstate__(self):
        state = super(Structured3, self).__getstate__()
        state['struct3'] = self.struct3
        return state

    def __setstate__(self, state):
        super(Structured3, self).__setstate__(state)
        self.struct3 = state['struct3']


# def test_convert_in():
#
#     multi = MultiType(TypeA, TypeB)
#
#     with pytest.raises(ValueError):
#         multi.convert_in(None)


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
    multi.current_node = multi.node_type[0]

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
    multi.current_node = multi.node_type[1]

    assert multi.tag_names() == ['domeB']


def test_internal_default():
    import numina.core.dataholders as dh

    multi = MultiType(TypeA, TypeB)
    req = dh.Requirement(multi, description="Some")

    with pytest.raises(ValueError):
        req.default_value()


def test_internal_default_optional():
    import numina.core.dataholders as dh

    multi = MultiType(TypeA, TypeB)
    req = dh.Requirement(multi, description="Some", optional=True)

    assert req.default_value() is None


def test_load_multitype(tmpdir):
    import numina.store
    import os
    import numina.util.context as cntx

    multi = MultiType(Structured1, Structured2)

    prefix1 = 'some1'
    prefix2 = 'some2'
    prefix3 = 'some3'

    filename1 = prefix1 + '.json'
    filename2 = prefix2 + '.json'
    filename3 = prefix3 + '.json'

    obj1 = Structured1()
    obj2 = Structured2()
    obj3 = Structured3()

    with cntx.working_directory(str(tmpdir)):
        numina.store.dump(obj1, obj1, prefix1)
        numina.store.dump(obj2, obj2, prefix2)
        numina.store.dump(obj3, obj3, prefix3)

    with cntx.working_directory(str(tmpdir)):

        recover = numina.store.load(multi, filename1)
        assert isinstance(recover, Structured1)

        recover = numina.store.load(multi, filename2)
        assert isinstance(recover, Structured2)

        with pytest.raises(TypeError):
            numina.store.load(multi, filename3)
