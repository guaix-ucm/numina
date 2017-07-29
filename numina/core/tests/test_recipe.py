#
# Copyright 2008-2017 Universidad Complutense de Madrid
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
from ..qc import QC


class PruebaRecipe1(BaseRecipe):
    somereq = Requirement(int, 'Some integer')
    someresult = Product(int, 'Some integer')
    qc = Product(QualityControlProduct)

    def run(self, recipe_input):
        if recipe_input.somereq >= 100:
            result = 1
        else:
            result = 100

        return self.create_result(someresult=result)

    def run_qc(self, recipe_input, recipe_result):
        if recipe_result.someresult >= 100:
            recipe_result.qc = QC.BAD
        else:
            recipe_result.qc = QC.GOOD

        return recipe_result


def test_metaclass_empty_base():

    class RecipeTest(with_metaclass(RecipeType, object)):
        pass

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeResult'


def test_metaclass():

    class RecipeTest(with_metaclass(RecipeType, object)):
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'


def test_recipe_with_autoqc():

    class RecipeTestAutoQC(BaseRecipe):
        qc82h = Product(QualityControlProduct, destination='qc')

    class RecipeTest(RecipeTestAutoQC):
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'

    assert 'qc' in RecipeTest.RecipeResult.stored()
    assert 'qc' in RecipeTest.products()

    for prod in RecipeTest.RecipeResult.stored().values():
        assert isinstance(prod, Product)

    qc = RecipeTest.RecipeResult.stored()['qc']

    assert isinstance(qc.type, QualityControlProduct)


def test_recipe_without_autoqc():

    class RecipeTest(BaseRecipe):
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'

    assert 'qc' not in RecipeTest.RecipeResult.stored()
    assert 'qc' not in RecipeTest.products()

    for prod in RecipeTest.RecipeResult.stored().values():
        assert isinstance(prod, Product)


def test_recipe_io_inheritance():

    class TestBaseRecipe(BaseRecipe):
        obresult = ObservationResultRequirement()
        someresult1 = Product(int, 'Some integer')

    class RecipeTest(TestBaseRecipe):
        other = Requirement(int, description='Other')
        someresult2 = Product(int, 'Some integer')

    assert issubclass(RecipeTest.RecipeInput, TestBaseRecipe.RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, TestBaseRecipe.RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'

    assert 'obresult' in RecipeTest.requirements()
    assert 'other' in RecipeTest.requirements()
    assert 'someresult1' in RecipeTest.products()
    assert 'someresult2' in RecipeTest.products()


def test_recipe_io_baseclass():

    class MyRecipeInput(RecipeInput):
        def myfunction(self):
            return 1

    class MyRecipeResult(RecipeResult):
        def myfunction(self):
            return 2

    class RecipeTest(BaseRecipe):

        RecipeInput = MyRecipeInput

        RecipeResult = MyRecipeResult

        other = Requirement(int, description='Other')
        someresult2 = Product(int, 'Some integer')

    assert issubclass(RecipeTest.RecipeInput, MyRecipeInput)

    assert issubclass(RecipeTest.RecipeResult, MyRecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'

    assert 'other' in RecipeTest.requirements()
    assert 'someresult2' in RecipeTest.products()

    assert RecipeTest.RecipeInput().myfunction() == 1
    assert RecipeTest.RecipeResult().myfunction() == 2


def test_run_qc():

    recipe_input = PruebaRecipe1.create_input(somereq=100)
    recipe = PruebaRecipe1()
    result = recipe(recipe_input)

    assert result.qc == QC.GOOD


def test_run_base():

    recipe_input = PruebaRecipe1.create_input(somereq=1)
    recipe = PruebaRecipe1()
    result = recipe(recipe_input)

    assert result.qc == QC.BAD
    assert result.someresult == 100

