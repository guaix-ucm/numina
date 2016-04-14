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
from ..recipes import BaseRecipe
from ..recipeinout import RecipeInput, RecipeResult
from ..requirements import ObservationResultRequirement
from ..dataholders import Product
from ..requirements import Requirement
from ..products import QualityControlProduct


def test_metaclass_empty_base():

    class TestRecipe(with_metaclass(RecipeType, object)):
        pass

    assert hasattr(TestRecipe, 'RecipeInput')

    assert hasattr(TestRecipe, 'RecipeResult')

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

    assert hasattr(TestRecipe, 'RecipeResult')

    assert TestRecipe.RecipeInput.__name__ == 'TestRecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'


def test_recipe_with_autoqc():

    class TestRecipeAutoQC(BaseRecipe):
        qc = Product(QualityControlProduct, dest='qc')

    class TestRecipe(TestRecipeAutoQC):
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(TestRecipe, 'RecipeInput')

    assert hasattr(TestRecipe, 'RecipeResult')

    assert issubclass(TestRecipe.RecipeInput, RecipeInput)

    assert issubclass(TestRecipe.RecipeResult, RecipeResult)

    assert TestRecipe.RecipeInput.__name__ == 'TestRecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'

    assert 'qc' in TestRecipe.RecipeResult.stored()
    assert 'qc' in TestRecipe.products()

    for prod in TestRecipe.RecipeResult.stored().values():
        assert isinstance(prod, Product)

    qc = TestRecipe.RecipeResult.stored()['qc']

    assert isinstance(qc.type, QualityControlProduct)


def test_recipe_without_autoqc():

    class TestRecipe(BaseRecipe):
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(TestRecipe, 'RecipeInput')

    assert hasattr(TestRecipe, 'RecipeResult')

    assert issubclass(TestRecipe.RecipeInput, RecipeInput)

    assert issubclass(TestRecipe.RecipeResult, RecipeResult)

    assert TestRecipe.RecipeInput.__name__ == 'TestRecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'

    assert 'qc' not in TestRecipe.RecipeResult.stored()
    assert 'qc' not in TestRecipe.products()

    for prod in TestRecipe.RecipeResult.stored().values():
        assert isinstance(prod, Product)


def test_recipe_io_inheritance():

    class TestBaseRecipe(BaseRecipe):
        obresult = ObservationResultRequirement()
        someresult1 = Product(int, 'Some integer')

    class TestRecipe(TestBaseRecipe):
        other = Requirement(int, description='Other')
        someresult2 = Product(int, 'Some integer')

    assert issubclass(TestRecipe.RecipeInput, TestBaseRecipe.RecipeInput)

    assert issubclass(TestRecipe.RecipeResult, TestBaseRecipe.RecipeResult)

    assert TestRecipe.RecipeInput.__name__ == 'TestRecipeInput'

    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'

    assert 'obresult' in TestRecipe.requirements()
    assert 'other' in TestRecipe.requirements()
    assert 'someresult1' in TestRecipe.products()
    assert 'someresult2' in TestRecipe.products()
