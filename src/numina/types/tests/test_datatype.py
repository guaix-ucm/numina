import pytest

from numina.exceptions import ValidationError
from numina.types.datatype import PlainPythonType


def validator_fun1(value):
    if value > 50:
        raise ValidationError
    return value


def test_no_check():
    value = 1
    obj = 2

    ppt = PlainPythonType(ref=value)
    assert ppt.convert(obj) == obj


def test_custom_check1():
    value = 1
    obj = 2

    ppt = PlainPythonType(ref=value, validator=validator_fun1)
    assert ppt.convert(obj) == obj


def test_custom_check2():
    value = 1
    obj = 200

    ppt = PlainPythonType(ref=value, validator=validator_fun1)
    with pytest.raises(ValidationError):
        ppt.convert(obj)


def test_conversion():
    value = 1.0
    obj1 = 2
    obj2 = 2.0

    ppt = PlainPythonType(ref=value, validator=validator_fun1)

    assert ppt.convert(obj1) == obj2
    assert isinstance(obj2, ppt.internal_type)
    assert ppt.internal_type is type(value)