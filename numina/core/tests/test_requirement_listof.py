
import pytest

import numina.core
import numina.core.query
import numina.dal.stored
import numina.exceptions
import numina.types.datatype as dt
import numina.tests.simpleobj
from numina.types.structured import BaseStructuredCalibration

from ..dataholders import Requirement


class Dal(object):
    def search_parameter(self, name, type_, obsres, options=None):
        return numina.dal.stored.StoredParameter(content=1)

    def search_product(self, name, type_, obsres, options=None):
        return numina.dal.stored.StoredProduct(id=1, content=2, tags={})


def test_list_of():

    req1 = Requirement(dt.ListOfType(BaseStructuredCalibration), description="ListOf", destination="mcalibs")
    req2 = Requirement(dt.PlainPythonType(ref=100), description="simple", destination="calibs")

    dal = Dal()

    obsres = numina.core.ObservationResult()
    obsres.tags = [{}, {}, {}]

    res = req1.query(dal, obsres)
    assert obsres.tags == [{}, {}, {}]
    assert res == [2, 2, 2]

    res = req2.query(dal, obsres)
    assert obsres.tags == [{}, {}, {}]
    assert res == 1
