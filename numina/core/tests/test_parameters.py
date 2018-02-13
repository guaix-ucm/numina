
import pytest

from ..requirements import Parameter
from numina.types.datatype import PlainPythonType
from numina.core.recipeinout import RecipeInput
from numina.exceptions import ValidationError

def test_requirement_in():

    value1 = 1
    value2 = 100

    class RR(RecipeInput):
        some = Parameter(value1, '')

    rr = RR()

    assert rr.some == value1

    class RR(RecipeInput):
        some = Parameter(value1, '')

    rr = RR(some=value2)

    assert rr.some == value2


def test_param_internal():

    value1 = 1

    class RR(RecipeInput):
        some = Parameter(value1, '')

    assert isinstance(RR.some, Parameter)
    assert isinstance(RR.some.type, PlainPythonType)


def validator_fun1(value):
    if value > 50:
        raise ValidationError
    return value


def test_param_check():

    value1 = 1
    value2 = 100
    value3 = 20

    class RR(RecipeInput):
        some = Parameter(value1, '', validator=validator_fun1)

    with pytest.raises(ValidationError):
        RR(some=value2)

    rr = RR(some=value3)

    assert rr.some == value3


def test_param_choices1():

    value1 = 1
    choices = [1,2,3,4]
    value2 = 2

    class RR(RecipeInput):
        some = Parameter(value1, '', choices=choices)

    rr = RR(some=value2)

    assert rr.some == value2


def test_param_choices2():

    value1 = 1
    choices = [1, 2, 3, 4]
    value2 = 8

    class RR(RecipeInput):
        some = Parameter(value1, '', choices=choices)

    with pytest.raises(ValidationError):
        RR(some=value2)


def test_param_dest():

    default = 34.3

    class RR(RecipeInput):
        some = Parameter(default, '', destination='other')

    assert hasattr(RR, 'other')
    assert not hasattr(RR, 'some')

    rr = RR()
    assert hasattr(rr, 'other')
    assert not hasattr(rr, 'some')

    rr = RR(other=default)
    assert hasattr(rr, 'other')
    assert not hasattr(rr, 'some')
