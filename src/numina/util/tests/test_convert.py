
import datetime
import pytest

from ..convert import convert_date


def test_convert_date1():
    result = datetime.datetime(2018, 2, 12, 22, 4, 30, 100000)
    value = convert_date("2018-02-12T22:04:30.1")
    assert value == result


def test_convert_date2():
    result = datetime.datetime(2018, 2, 12, 22, 4, 30)
    value = convert_date("2018-02-12T22:04:30")
    assert value == result


def test_convert_date3():
    result = datetime.datetime(2017, 6, 27, 0, 0)
    value = convert_date("2017-06-27")
    assert value == result


def test_convert_date4():
    with pytest.raises(ValueError):
        convert_date("something")
