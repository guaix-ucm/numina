#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Unit test for RecipeBase."""

from numina.core.recipes import BaseRecipe
from numina.core.recipeinout import RecipeInput, RecipeResult
from numina.core.requirements import ObservationResultRequirement
from numina.core.dataholders import Result
from numina.core.requirements import Requirement
from numina.types.qc import QC
from numina.types.datatype import PlainPythonType


class PruebaRecipe1(BaseRecipe):
    somereq = Requirement(int, "Some integer")
    someresult = Result(int, "Some integer")

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


def test_recipe_empty_base():

    class RecipeTest(BaseRecipe):
        pass

    assert hasattr(RecipeTest, "RecipeInput")

    assert hasattr(RecipeTest, "RecipeResult")

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == "RecipeInput"

    assert RecipeTest.RecipeResult.__name__ == "RecipeResult"


def test_recipe_io_classes():

    class RecipeTest(BaseRecipe):
        obsresult = ObservationResultRequirement()
        someresult = Result(int, "Some integer")

    assert hasattr(RecipeTest, "RecipeInput")

    assert hasattr(RecipeTest, "RecipeResult")

    assert RecipeTest.RecipeInput.__name__ == "RecipeTestInput"

    assert RecipeTest.RecipeResult.__name__ == "RecipeTestResult"


def test_recipe_io_classes_inherit():

    class RecipeBase1(BaseRecipe):
        req1 = Requirement(int, "Some integer")
        res1 = Result(int, "Some integer")

    class RecipeTest(RecipeBase1):
        req2 = Requirement(int, "Other integer")

    # the requirements and results are moved to the generated classes
    assert not hasattr(RecipeTest, "req1")
    assert not hasattr(RecipeTest, "req2")
    assert list(RecipeTest.requirements()) == ["req1", "req2"]
    assert list(RecipeTest.products()) == ["res1"]
    assert issubclass(RecipeTest.RecipeInput, RecipeBase1.RecipeInput)
    # without new results, the result class of the base recipe is used
    assert RecipeTest.RecipeResult is RecipeBase1.RecipeResult
    assert RecipeTest.RecipeTestInput is RecipeTest.RecipeInput
    assert RecipeTest.RecipeInput.__module__ == __name__
    assert RecipeTest.RecipeInput.__qualname__.endswith("RecipeTest.RecipeTestInput")


def test_recipe_init_without_arguments():

    class RecipeTest(BaseRecipe):
        __version__ = "class version"

        def __init__(self, *args, **kwargs):
            # the arguments are not passed
            super().__init__()

    recipe = RecipeTest(instrument="TEST", mode="bias", version="2", runinfo={"taskid": "1"})
    assert recipe.instrument == "TEST"
    assert recipe.mode == "bias"
    assert recipe.__version__ == "2"
    assert recipe.runinfo["taskid"] == "1"
    assert recipe.runinfo["pipeline"] == "default"


def test_recipe_version():

    class RecipeDefault(BaseRecipe):
        pass

    class RecipeClassVersion(BaseRecipe):
        __version__ = "3"

    assert RecipeDefault().__version__ == 1
    # the version of the class is used, it was always 1
    assert RecipeClassVersion().__version__ == "3"
    assert RecipeClassVersion(version="4").__version__ == "4"

    hdr = RecipeClassVersion().set_base_headers({})
    assert hdr["NUMRVER"] == ("3", "Numina recipe version")


def test_recipe_with_autofield():

    class RecipeTestAutoField(BaseRecipe):
        qc82h = Result(float, destination="qc")

    class RecipeTest(RecipeTestAutoField):
        obsresult = ObservationResultRequirement()
        someresult = Result(int, "Some integer")

    assert hasattr(RecipeTest, "RecipeInput")

    assert hasattr(RecipeTest, "RecipeResult")

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == "RecipeTestInput"

    assert RecipeTest.RecipeResult.__name__ == "RecipeTestResult"

    assert "qc" in RecipeTest.RecipeResult.stored()
    assert "qc" in RecipeTest.products()

    for prod in RecipeTest.RecipeResult.stored().values():
        assert isinstance(prod, Result)

    qc = RecipeTest.RecipeResult.stored()["qc"]

    assert isinstance(qc.type, PlainPythonType)


def test_recipe_without_autofield():

    class RecipeTest(BaseRecipe):
        obsresult = ObservationResultRequirement()
        someresult = Result(int, "Some integer")

    assert hasattr(RecipeTest, "RecipeInput")

    assert hasattr(RecipeTest, "RecipeResult")

    assert issubclass(RecipeTest.RecipeInput, RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == "RecipeTestInput"

    assert RecipeTest.RecipeResult.__name__ == "RecipeTestResult"

    assert "qc" not in RecipeTest.RecipeResult.stored()
    assert "qc" not in RecipeTest.products()

    for prod in RecipeTest.RecipeResult.stored().values():
        assert isinstance(prod, Result)


def test_recipe_io_inheritance():

    class TestBaseRecipe(BaseRecipe):
        obresult = ObservationResultRequirement()
        someresult1 = Result(int, "Some integer")

    class RecipeTest(TestBaseRecipe):
        other = Requirement(int, description="Other")
        someresult2 = Result(int, "Some integer")

    assert issubclass(RecipeTest.RecipeInput, TestBaseRecipe.RecipeInput)

    assert issubclass(RecipeTest.RecipeResult, TestBaseRecipe.RecipeResult)

    assert RecipeTest.RecipeInput.__name__ == "RecipeTestInput"

    assert RecipeTest.RecipeResult.__name__ == "RecipeTestResult"

    assert "obresult" in RecipeTest.requirements()
    assert "other" in RecipeTest.requirements()
    assert "someresult1" in RecipeTest.products()
    assert "someresult2" in RecipeTest.products()


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

        other = Requirement(int, description="Other")
        someresult2 = Result(int, "Some integer")

    assert issubclass(RecipeTest.RecipeInput, MyRecipeInput)

    assert issubclass(RecipeTest.RecipeResult, MyRecipeResult)

    assert RecipeTest.RecipeInput.__name__ == "RecipeTestInput"

    assert RecipeTest.RecipeResult.__name__ == "RecipeTestResult"

    assert "other" in RecipeTest.requirements()
    assert "someresult2" in RecipeTest.products()

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
