#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#


"""Unit test for RecipeBase."""

from six import with_metaclass

from ..metarecipes import RecipeType
from ..recipes import BaseRecipeAutoQC
from ..recipeinout import RecipeInput, RecipeResult
from ..requirements import ObservationResultRequirement
from ..dataholders import Product
from ..products import QualityControlProduct


def test_metaclass_empty_base():

    class TestRecipe(with_metaclass(RecipeType, object)):
        pass

    assert hasattr(TestRecipe, 'RecipeInput')

    assert hasattr(TestRecipe, 'RecipeResult')

    assert hasattr(TestRecipe, 'Input')

    assert hasattr(TestRecipe, 'Result')

    assert issubclass(TestRecipe.RecipeInput, RecipeInput)

    assert issubclass(TestRecipe.RecipeResult, RecipeResult)

    assert TestRecipe.RecipeInput.__name__ == 'RecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'RecipeResult'


def test_metaclass():

    class TestRecipe(with_metaclass(RecipeType, object)):
        pass   

        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(TestRecipe, 'RecipeInput')
    assert hasattr(TestRecipe, 'Input')

    assert hasattr(TestRecipe, 'RecipeResult')
    assert hasattr(TestRecipe, 'Result')

    assert issubclass(TestRecipe.Input, RecipeInput)

    assert issubclass(TestRecipe.Result, RecipeResult)

    assert TestRecipe.RecipeInput.__name__ == 'TestRecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'


def test_recipe_with_autoqc():

    class TestRecipe(BaseRecipeAutoQC):
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(TestRecipe, 'RecipeInput')

    assert hasattr(TestRecipe, 'RecipeResult')

    assert hasattr(TestRecipe, 'Input')

    assert hasattr(TestRecipe, 'Result')

    assert issubclass(TestRecipe.RecipeInput, RecipeInput)

    assert issubclass(TestRecipe.RecipeResult, RecipeResult)

    assert TestRecipe.RecipeInput.__name__ == 'TestRecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'

    assert 'qc' in TestRecipe.RecipeResult

    for prod in TestRecipe.RecipeResult.values():
        assert isinstance(prod, Product)

    qc = TestRecipe.RecipeResult['qc']

    assert isinstance(qc.type, QualityControlProduct)
