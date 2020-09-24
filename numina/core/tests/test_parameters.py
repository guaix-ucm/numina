
import pytest

from ..dataholders import Parameter

from numina.core.validator import as_list, only_positive
from numina.types.datatype import PlainPythonType
from numina.core.recipeinout import RecipeInput
from numina.exceptions import ValidationError
from numina.types.datatype import ListOfType


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


def test_param_accept_scalar():
    """Test accept_scalar argument"""
    some = Parameter([1], 'Convert scalar to list', accept_scalar=True)
    result = some.convert(1)
    assert result == [1]

    assert isinstance(some.type, ListOfType)
    assert isinstance(some.type.node_type, PlainPythonType)
    assert some.type.internal_type is list


def test_param_as_list1():
    """Test as_list argument"""
    some = Parameter(1, 'Convert scalar to list', as_list=True)
    result = some.convert([1, 2, 3])
    assert result == [1, 2, 3]

    result = some.convert(1)
    assert result == [1]


def test_param_as_list2():
    """Test list of tuples"""
    some = Parameter([(0, 0)], 'List of coordinates')
    result = some.convert([(3, 4), (1, 1)])
    assert result == [[3,4], [1,1]]


@pytest.mark.xfail
def test_param_as_list3():
    """Test list of tuples"""
    some = Parameter([(0, 0)], 'List of coordinates')
    result = some.convert([(3, 4), (1, 1)])
    assert result == [(3,4), (1,1)]


@pytest.mark.parametrize("nelem, allowed, not_allowed", [
    ('*', [[], [1], [2,3,4]], []),
    ('+', [[1], [2, 3, 4]], [[]]),
    (1, [[1]], [[], [3.1, 34.0, 4]])
])
def test_param_as_list4(nelem, allowed, not_allowed):
    """Test list of tuples"""
    some = Parameter(5, 'List of coordinates', nelem=nelem)
    for obj in allowed:
        assert obj == some.convert(obj)

    for obj in not_allowed:
        with pytest.raises(ValidationError):
            some.convert(obj)


def test_param_custom_validator1():
    """Test accept_scalar argument"""
    some = Parameter(1, 'Validate', validator=only_positive)

    with pytest.raises(ValidationError):
        some.convert(-11)


def test_param_custom_validator2():
    """Test accept_scalar argument"""
    some = Parameter(1, 'Validate', as_list=True, validator=only_positive)

    with pytest.raises(ValidationError):
        some.convert(-11)

    with pytest.raises(ValidationError):
        some.convert([-11])


def test_param_custom_validator3():
    """Test accept_scalar argument"""
    some = Parameter([1], 'Validate', validator=as_list(only_positive))

    with pytest.raises(ValidationError):
        some.convert([-33])


def test_param_custom_validator4():
    """Test accept_scalar argument"""
    some = Parameter([1], 'Validate', accept_scalar=True,
                     validator=as_list(only_positive))

    with pytest.raises(ValidationError):
        some.convert(-11)

    with pytest.raises(ValidationError):
        some.convert([50, -22])


def test_param_default1():
    """Test accept_scalar argument"""
    some = Parameter([1], 'Validate')

    assert some.default_value() == [1]


def test_param_default2():
    """Test accept_scalar argument"""
    some = Parameter(1, 'Validate', as_list=True)

    assert some.default_value() == [1]


def test_empty_list_escalar():
    some = Parameter([], 'Some', accept_scalar=True)
    value = some.convert(1)
    assert value == [1]


def test_empty_list_iter():
    some = Parameter([], 'Some')
    value = some.convert([2])
    assert value == [2]

    value = some.convert(["A", 34.2])
    assert value == ["A", 34.2]


def test_param_bool1():
    """Test bool argument"""

    value1 = True
    value2 = False

    class RR(RecipeInput):
        some = Parameter(value1, '')

    rr = RR()

    assert rr.some == value1

    class RR(RecipeInput):
        some = Parameter(value2, '')

    rr = RR(some=value2)

    assert rr.some == value2
