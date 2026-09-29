import numina.core.pipeline

import pytest

from ..drpbase import DrpBase


def test_drpbase():
    drpbase = DrpBase()

    with pytest.raises(KeyError):
        drpbase.query_by_name("TEST1")

    assert drpbase.query_all() == {}


def test_invalid_instrument1():

    class Something(object):
        pass

    drpbase = DrpBase()
    with pytest.warns(RuntimeWarning, match="does not contain a valid DRP"):
        assert drpbase.instrumentdrp_check(Something(), "TEST1") is False


def test_invalid_instrument2():
    insdrp = numina.core.pipeline.InstrumentDRP("MYNAME", {}, {}, [], [])

    drpbase = DrpBase()
    with pytest.warns(RuntimeWarning, match="differ"):
        res = drpbase.instrumentdrp_check(insdrp, "TEST1")
    assert res is False


def test_valid_instrument():
    insdrp = numina.core.pipeline.InstrumentDRP("TEST1", {}, {}, [], [])

    drpbase = DrpBase()
    res = drpbase.instrumentdrp_check(insdrp, "TEST1")
    assert res
