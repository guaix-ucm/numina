
from ..configorigin import UndefinedOrigin,ElementOrigin

import pytest


def test_undefined_origin():
    origin = UndefinedOrigin()
    assert origin.is_valid_date(None)


def _origin1():
    date_start = "2001-10-10T12:00:00"
    date_end = None
    name = 'origin'
    cuuid = 'c27287cd-90d1-417c-8238-f5ff220f818d'
    origin = ElementOrigin(
        name, cuuid,
        date_start=date_start,
        date_end=date_end
    )
    return origin


def _origin2():
    date_start = "2001-10-10T12:00:00"
    date_end = "2002-10-10T12:00:00"
    name = 'origin'
    cuuid = 'c27287cd-90d1-417c-8238-f5ff220f818d'
    origin = ElementOrigin(
        name, cuuid,
        date_start=date_start,
        date_end=date_end
    )
    return origin


_test_vals1 = [
    (None, True),
    ("2001-10-09T12:00:00", False),
    ("2001-10-11T12:00:00", True),
    ("2002-10-11T12:00:00", True),
    ("2002-10-10T12:00:00", True),
    ("2005-10-10T12:00:00", True),
]

_test_vals2 = [
    (None, True),
    ("2001-10-09T12:00:00", False),
    ("2001-10-10T12:00:00", True),
    ("2002-02-11T12:00:00", True),
    ("2002-10-10T12:00:00", False),
    ("2005-10-10T12:00:00", False),
]

@pytest.mark.parametrize("origin", [_origin1()])
@pytest.mark.parametrize("date, result", _test_vals1)
def test_origin1(origin, date, result):
    calc = origin.is_valid_date(date)
    assert result == calc


@pytest.mark.parametrize("origin", [_origin2()])
@pytest.mark.parametrize("date, result", _test_vals2)
def test_origin2(origin, date, result):
    calc = origin.is_valid_date(date)
    assert result == calc


def test_create_keys():
    date_start = "2001-10-10T12:00:00"
    date_end = "2002-10-10T12:00:00"
    name = 'origin'
    cuuid = 'c27287cd-90d1-417c-8238-f5ff220f818d'

    origin = ElementOrigin.create_from_keys(
        name=name,
        uuid=cuuid,
        date_start=date_start,
        date_end=date_end
    )
    assert isinstance(origin, ElementOrigin)
    assert origin.name == name
    assert str(origin.uuid) == cuuid
    assert origin.date_start.isoformat() == date_start
    assert origin.date_end.isoformat() == date_end
    # assert origin.description == description

def test_create_dict():
    values = dict(
        date_start="2001-10-10T12:00:00",
        date_end="2002-10-10T12:00:00",
        name='origin',
        uuid='c27287cd-90d1-417c-8238-f5ff220f818d'
    )

    origin = ElementOrigin.create_from_dict(values)
    assert isinstance(origin, ElementOrigin)
    assert origin.name == values['name']
    assert str(origin.uuid) == values['uuid']
    assert origin.date_start.isoformat() == values['date_start']
    assert origin.date_end.isoformat() == values['date_end']
    assert origin.description == values.get('description', '')