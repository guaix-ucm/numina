#
# Copyright 2015-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

import numina.types.qc as qct
import numina.types.datatype as df
import numina.core.dataholders as dh

from ..recipeinout import RecipeResult, RecipeInput


class RRTest(RecipeResult):
    param1 = dh.Result(df.NullType, 'something1')
    param2 = dh.Result(df.NullType, 'something2')

    def mayfun(self):
        pass


def test_test1():

    m = RecipeResult()
    assert hasattr(m, 'qc')


def test_test2():
    m = RecipeResult()
    assert m.qc == qct.QC.UNKNOWN


def test_test3():
    m = RRTest(param1=None, param2=None)
    assert m.qc == qct.QC.UNKNOWN

    m = RRTest(param1=None, param2=None, qc=qct.QC.GOOD)
    assert m.qc == qct.QC.GOOD


@pytest.mark.parametrize("qc", [
    qct.QC.GOOD,
    qct.QC.UNKNOWN,
    qct.QC.BAD,
    qct.QC.PARTIAL,
])
def test_test4(qc):

    m = RRTest(param1=None, param2=None, qc=qc)
    assert m.qc == qc


@pytest.mark.parametrize("qc", [
    qct.QC.GOOD,
    qct.QC.UNKNOWN,
    qct.QC.BAD,
    qct.QC.PARTIAL,
])
def test_store_to(qc):
    result = RRTest(param1=None, param2=None, qc=qc)

    expected_result = {
        'values': {'param1': None, 'param2': None},
        'qc': qc.name, # 'uuid': '00000000-0000-0000-0000-000000000000'
    }

    class Storage(object):
        def __init__(self):
            self.runinfo = {}

    where = Storage()

    saved_result = result.store_to(where)

    assert 'uuid' in saved_result

    # Do not check uuid field
    del saved_result['uuid']
    assert saved_result == expected_result


def test_capture_conversion_error():

    class RRTest1(RecipeInput):
        param1 = dh.Parameter(value=[100], description='some1')

    with pytest.raises(ValueError):
        RRTest1(param1=100)
