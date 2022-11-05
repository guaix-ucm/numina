
import pytest

import numina.core
import numina.core.query
import numina.dal.stored
import numina.exceptions
import numina.types.datatype as dt
import numina.tests.simpleobj
from numina.types.structured import BaseStructuredCalibration
from numina.types.multitype import MultiType
from ..dataholders import Requirement


class One(BaseStructuredCalibration):
    pass


class Other(BaseStructuredCalibration):
    pass


class Dal(object):
    def search_parameter(self, name, stype, obsres, options=None):
        if name == 'req5':
            return numina.dal.stored.StoredParameter(content=[1, 300, 12])
        else:
            return numina.dal.stored.StoredParameter(content=1)

    def search_product(self, name, stype, obsres, options=None):
        if name == 'req4':
            if stype.name() == 'One' and obsres.tags == {'a': 1, 'b': 2}:
                return numina.dal.stored.StoredProduct(id=1, content=11, tags={})
            elif stype.name() == 'One' and obsres.tags == {'a': 2, 'b':1}:
                return numina.dal.stored.StoredProduct(id=10, content=12, tags={})
            elif stype.name() == 'Other':
                return numina.dal.stored.StoredProduct(id=100, content=24, tags={})
            else:
                raise numina.exceptions.NoResultFound(f'{stype.name()} not found')
        else:
            return numina.dal.stored.StoredProduct(id=1, content=2, tags={})



def test_list_of():

    req1 = Requirement(dt.ListOfType(One), description="ListOf", destination="req1")
    req2 = Requirement(dt.PlainPythonType(ref=100), description="simple", destination="req2")
    req3 = Requirement(MultiType(One, Other), description="simple", destination="req3")
    req4 = Requirement(dt.ListOfType(MultiType(One, Other)), description="simple", destination="req4")
    req5 = Requirement(dt.ListOfType(dt.PlainPythonType(ref=100)), description="simple", destination="req5")

    dal = Dal()

    obsres = numina.core.ObservationResult()
    obsres.tags = [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]

    assert obsres.tags == [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]
    res1 = req1.query(dal, obsres) # P [2, 2, 2]
    assert obsres.tags == [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]
    res2 = req2.query(dal, obsres) #   1
    assert obsres.tags == [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]
    res3 = req3.query(dal, obsres) # P 2
    assert obsres.tags == [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]
    res4 = req4.query(dal, obsres) # P [2,2 ,2 ]
    assert obsres.tags == [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]
    res5 = req5.query(dal, obsres) #   [1,300,12]
    assert obsres.tags == [{'a': 1, 'b':2}, {'a':2, 'b':1}, {'a':2, 'b': 2}]

    assert res1 == [2, 2, 2]
    assert res2 == 1
    assert res3 == 2
    assert res4 == [11, 12, 24]
    assert res5 == [1, 300, 12]
