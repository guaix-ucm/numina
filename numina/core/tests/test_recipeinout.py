#
# Copyright 2015-2018 Universidad Complutense de Madrid
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

from ..recipeinout import RecipeResultQC, RecipeInput


class RRTest(RecipeResultQC):
    param1 = dh.Result(df.NullType, 'something1')
    param2 = dh.Result(df.NullType, 'something2')

    def mayfun(self):
        pass


def test_test1():

    m = RecipeResultQC()
    assert hasattr(m, 'qc')


def test_test2():
    m = RecipeResultQC()
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
    m = RRTest(param1=None, param2=None, qc=qc)

    exp = {'values': {'param1': None, 'param2': None}, 'qc': qc}

    class Storage(object):
        def __init__(self):
            self.runinfo = {}

    where = Storage()

    saveres = m.store_to(where)
    assert saveres == exp


def test_capture_conversion_error():

    class RRTest1(RecipeInput):
        param1 = dh.Parameter(value=[100], description='some1')

    with pytest.raises(ValueError):
        RRTest1(param1=100)
