#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Unit test for RecipeBase."""

from six import with_metaclass

from ..metarecipes import RecipeType
from ..recipes import BaseRecipe
from ..recipeinout import RecipeInput, RecipeResult
from ..requirements import ObservationResultRequirement
from ..dataholders import Result
from ..requirements import Requirement
from numina.types.obsresult import QualityControlProduct
from numina.types.qc import QC


class PruebaRecipe1(BaseRecipe):
    somereq = Requirement(int, 'Some integer')
    someresult = Result(int, 'Some integer')

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
        someresult = Result(int, 'Some integer')

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'


def test_recipe_with_autoqc():

    class RecipeTestAutoQC(BaseRecipe):
        qc82h = Result(QualityControlProduct, destination='qc')

    class RecipeTest(RecipeTestAutoQC):
        obsresult = ObservationResultRequirement()
        someresult = Result(int, 'Some integer')

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'

    assert 'qc' in RecipeTest.RecipeResult.stored()
    assert 'qc' in RecipeTest.products()

    for prod in RecipeTest.RecipeResult.stored().values():
        assert isinstance(prod, Result)

    qc = RecipeTest.RecipeResult.stored()['qc']

    assert isinstance(qc.type, QualityControlProduct)


def test_recipe_without_autoqc():

    class RecipeTest(BaseRecipe):
        obsresult = ObservationResultRequirement()
        someresult = Result(int, 'Some integer')

    assert hasattr(RecipeTest, 'RecipeInput')

    assert hasattr(RecipeTest, 'RecipeResult')

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == 'RecipeTestInput'

    assert RecipeTest.RecipeResult.__name__ == 'RecipeTestResult'

    assert 'qc' not in RecipeTest.RecipeResult.stored()
    assert 'qc' not in RecipeTest.products()

    for prod in RecipeTest.RecipeResult.stored().values():
        assert isinstance(prod, Result)


def test_recipe_io_inheritance():

    class TestBaseRecipe(BaseRecipe):
        obresult = ObservationResultRequirement()
        someresult1 = Result(int, 'Some integer')

    class RecipeTest(TestBaseRecipe):
        other = Requirement(int, description='Other')
        someresult2 = Result(int, 'Some integer')

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
        someresult2 = Result(int, 'Some integer')

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

