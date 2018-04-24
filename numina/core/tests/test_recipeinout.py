#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

import pytest

import numina.types.qc as qct
import numina.types.datatype as df
import numina.core.dataholders as dh

from ..recipeinout import RecipeResultQC


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

    exp = {'param1': None, 'param2': None, 'qc': qc}

    class Storage(object):
        pass

    where = Storage()

    saveres = m.store_to(where)

    assert saveres == exp
