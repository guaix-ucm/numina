import pytest

import numina.exceptions
import numina.core.recipes
from ..validator import validate


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