
import pytest

import numina.core
import numina.core.query
import numina.dal.stored
import numina.exceptions
import numina.types.datatype

from ..dataholders import Requirement


class Dal(object):
    def search_parameter(self, name, type_, obsres, options=None):
        if name == 'req_null':
            return numina.dal.stored.StoredParameter(content=None)
        elif name == 'req_int':
            return numina.dal.stored.StoredParameter(content=2)
        else:
            raise numina.exceptions.NoResultFound('value not found')


def test_query1():

    req = Requirement(rtype=None, description="", destination='req_null')

    dal = Dal()

    obres = numina.core.ObservationResult()

    result = req.query(dal, obres)
    assert result is None


def test_query11():

    req = Requirement(rtype=numina.types.datatype.PlainPythonType(1), description="", destination='req_int')

    dal = Dal()

    obres = numina.core.ObservationResult()

    result = req.query(dal, obres)
    assert result == 2
    # assert False


def test_dest_not_set():

    req = Requirement(rtype=None, description="")

    dal = Dal()

    obres = numina.core.ObservationResult()

    with pytest.raises(ValueError):
        req.query(dal, obres)
