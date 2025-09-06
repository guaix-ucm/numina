import pytest

from ..configorigin import ElementOrigin
from ..elements import ElementEnum, ElementBase


@pytest.mark.parametrize(
    "name, value",
    [
        ("instrument", ElementEnum.ELEM_INSTRUMENT),
        ("component", ElementEnum.ELEM_COMPONENT),
        ("properties", ElementEnum.ELEM_PROPERTIES),
        ("setup", ElementEnum.ELEM_SETUP),
    ],
)
def test_from_str(name, value):
    assert ElementEnum.from_str(name) == value


def test_from_str_err():
    with pytest.raises(ValueError):
        assert ElementEnum.from_str("xxxx")


def test_origin():
    origin = ElementOrigin("Org", "87158f24-6cea-4851-a197-d6515e0592f5")
    a = ElementBase("elem")
    a.set_origin(origin)
    assert a.origin == origin
