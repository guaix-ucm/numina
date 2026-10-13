"""DataTypeBase.from_name: a datatype from its name"""

import pytest

from numina.types.datatype import PlainPythonType
from numina.types.linescatalog import LinesCatalog


class OtherLinesCatalog(LinesCatalog):
    """A subclass of a DRP that keeps the name of the generic type"""

    def name(self):
        return "LinesCatalog"


def test_from_class_name():
    assert isinstance(LinesCatalog.from_name("LinesCatalog"), LinesCatalog)


def test_from_class_name_with_arguments():
    obj = PlainPythonType.from_name("PlainPythonType(ref=1)")
    assert isinstance(obj, PlainPythonType)
    assert obj.internal_type is int


def test_from_datatype_name():
    """The name of the datatype is accepted, if it is not the name of the class"""
    obj = OtherLinesCatalog.from_name("LinesCatalog")
    assert type(obj) is OtherLinesCatalog
    assert isinstance(OtherLinesCatalog.from_name("OtherLinesCatalog"), OtherLinesCatalog)


def test_from_other_name():
    with pytest.raises(TypeError, match="Other is not a name of LinesCatalog"):
        LinesCatalog.from_name("Other")
