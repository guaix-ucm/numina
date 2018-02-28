import pytest

import numina.exceptions
import numina.core.recipes
from ..validator import validate
from ..validator import only_positive
from ..validator import as_list
from ..validator import range_validator


class RecipeIO(object):
    def __init__(self, valid=True):
        self.valid = valid
        self.was_called = False

    def validate(self):
        self.was_called = True
        if not self.valid:
            raise numina.exceptions.ValidationError("This value is invalid")
        return True


class RecipeInput(RecipeIO):
    pass


class RecipeResult(RecipeIO):
    pass


class Recipe1(numina.core.recipes.BaseRecipe):
    @validate
    def run(self, rinput):
        return RecipeResult()


class Recipe2(numina.core.recipes.BaseRecipe):
    @validate
    def run(self, rinput):
        return RecipeResult(valid=False)


def test_validate():

    recipe = Recipe1()
    rinput = RecipeInput()
    result = recipe.run(rinput)

    assert rinput.was_called
    assert result.was_called


def test_validate_exception1():

    recipe = Recipe1()
    rinput = RecipeInput(valid=False)
    with pytest.raises(numina.exceptions.ValidationError):
        recipe.run(rinput)


def test_validate_exception2():

    recipe = Recipe2()
    rinput = RecipeInput(valid=True)
    with pytest.raises(numina.exceptions.ValidationError):
        recipe.run(rinput)


def test_only_positive1():
    """Test only_positive pass positice values"""
    for value in [0, 1, 2, 50, 400]:
        assert value == only_positive(value)


def test_only_positive2():
    """Test only_positive raises on negative values"""
    for value in [-1, -2, -50, -400, -1.3, -2939]:
        with pytest.raises(numina.exceptions.ValidationError):
            only_positive(value)


def test_as_list1():
    """Test as_list decorator"""
    @as_list
    def some_test(value):
        return value

    values = [0, 1, 2, 50, 400]
    assert values == some_test(values)


def test_as_list2():
    """Test as_list decorator raises ValidationError"""
    @as_list
    def some_test(value):
        if value > 1:
            raise numina.exceptions.ValidationError('must be <= 1')
        return value

    values = [0, 1, 2, 50, 400]
    with pytest.raises(numina.exceptions.ValidationError):
        some_test(values)


@pytest.mark.parametrize("validator, allowed, not_allowed", [
    (range_validator(), [-1, 2.0, 34.0, 4, -5], []),
    (range_validator(minval=3), [3, 34.0, 4], [2.99, -1]),
    (range_validator(maxval=3), [3, -1, -100], [3.1, 34.0, 4]),
    (range_validator(minval=2, maxval=3), [2, 2.5, 3], [-4, -1, 100])
])
def test_range_validator3(validator, allowed, not_allowed):
    "Range validators"

    for val in allowed:
        assert val == validator(val)

    for val in not_allowed:
        with pytest.raises(numina.exceptions.ValidationError):
            validator(val)
