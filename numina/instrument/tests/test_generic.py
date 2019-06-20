import pytest

from ..generic import ComponentGeneric
from ..configorigin import ElementOrigin


def test_uuid1():

    comp = ComponentGeneric('testcomp')
    with pytest.raises(AttributeError):
        assert comp.uuid == 0


def test_uuid2():

    comp = ComponentGeneric('testcomp')
    uuid_str = '8e87de0f-2584-4618-9bdb-76c6420f26cf'
    origin = ElementOrigin('testconf', uuid_str)
    comp.set_origin(origin)

    assert str(comp.uuid) == uuid_str
    assert str(comp.origin.uuid) == uuid_str
