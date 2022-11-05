import numina.core.pipeline

import pytest

from ..drpbase import DrpBase


def test_drpbase():
    drpbase = DrpBase()

    with pytest.raises(KeyError):
        drpbase.query_by_name('TEST1')

    assert drpbase.query_all() == {}


def test_invalid_instrument1():

    class Something(object):
        pass

    drpbase = DrpBase()
    assert drpbase.instrumentdrp_check(Something(), 'TEST1') == False


def test_invalid_instrument1_warning():

    with pytest.warns(RuntimeWarning):
        test_invalid_instrument1()


def test_invalid_instrument2():
    insdrp = numina.core.pipeline.InstrumentDRP('MYNAME', {}, {}, [], [])

    drpbase = DrpBase()
    res = drpbase.instrumentdrp_check(insdrp, 'TEST1')
    assert res == False


@pytest.mark.xfail(reason="warning seems unreliable")
def test_invalid_instrument2_warning():

    insdrp = numina.core.pipeline.InstrumentDRP('MYNAME', {}, {}, [], [])

    drpbase = DrpBase()
    with pytest.warns(RuntimeWarning):
        drpbase.instrumentdrp_check(insdrp, 'TEST1')


def test_valid_instrument():
    insdrp = numina.core.pipeline.InstrumentDRP('TEST1', {}, {}, [], [])

    drpbase = DrpBase()
    res = drpbase.instrumentdrp_check(insdrp, 'TEST1')
    assert res