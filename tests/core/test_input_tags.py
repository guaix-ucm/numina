"""The tags of the products of the input of a recipe, checked with TagsChecker"""

import pytest

from numina.core import BaseRecipe, ObservationResult, Requirement
from numina.core.recipeinout import RecipeInput, TagsChecker
from numina.core.requirements import ObservationResultRequirement
from numina.exceptions import ValidationError
from numina.keydef import QueryAttribute
from numina.types.multitype import MultiType
from numina.types.structured import BaseStructuredCalibration

from tests.support.drpdef import make_drp


class TracesMap(BaseStructuredCalibration):
    """A calibration with the tags vph and insmode, as the products of MEGARA"""

    __tags__ = {"vph": QueryAttribute("vph", str), "insmode": QueryAttribute("insmode", str)}


class ModelMap(BaseStructuredCalibration):
    __tags__ = {"vph": QueryAttribute("vph", str)}


class CheckedRecipeInput(RecipeInput):
    """The input of the recipes of an instrument, with the check of the tags"""

    __checkers__ = [TagsChecker()]


class TracedRecipe(BaseRecipe):
    RecipeInput = CheckedRecipeInput
    obresult = ObservationResultRequirement()
    master_traces = Requirement(TracesMap, "Traces")
    master_apertures = Requirement(MultiType(ModelMap, TracesMap), "Apertures", optional=True)


@pytest.fixture(autouse=True)
def drp_test1(drpmocker):
    """The DRP of TEST1, used to validate the observation result"""
    drpmocker.add_drp("TEST1", make_drp({"traced": TracedRecipe}))


def calibration(cls, **tags):
    obj = cls(instrument="TEST1")
    obj.tags = tags
    return obj


def recipe_input(traces, apertures=None):
    ob = ObservationResult(instrument="TEST1", mode="traced")
    ob.tags = {"vph": "HR-R", "insmode": "LCB"}
    return TracedRecipe().create_input(obresult=ob, master_traces=traces, master_apertures=apertures)


def test_tags_match():
    TracedRecipe().validate_input(recipe_input(calibration(TracesMap, vph="HR-R", insmode="LCB")))


def test_tags_do_not_match():
    rinput = recipe_input(calibration(TracesMap, vph="LR-U", insmode="LCB"))
    msg = "requirement master_traces: the value of master_traces has vph=LR-U, the observing block has vph=HR-R"
    with pytest.raises(ValidationError, match=msg):
        TracedRecipe().validate_input(rinput)


@pytest.mark.parametrize("cls, tags", [(ModelMap, {"vph": "HR-R"}), (TracesMap, {"vph": "HR-R", "insmode": "LCB"})])
def test_tags_of_one_of_the_types(cls, tags):
    """A product of any of the types of a MultiType"""
    traces = calibration(TracesMap, vph="HR-R", insmode="LCB")
    TracedRecipe().validate_input(recipe_input(traces, calibration(cls, **tags)))


def test_tags_of_none_of_the_types():
    traces = calibration(TracesMap, vph="HR-R", insmode="LCB")
    rinput = recipe_input(traces, calibration(ModelMap, vph="LR-U"))
    with pytest.raises(ValidationError, match="requirement master_apertures"):
        TracedRecipe().validate_input(rinput)


def test_tags_not_checked_by_default():
    """Without the checker, the tags are not checked"""

    class UncheckedRecipe(BaseRecipe):
        obresult = ObservationResultRequirement()
        master_traces = Requirement(TracesMap, "Traces")

    ob = ObservationResult(instrument="TEST1", mode="traced")
    ob.tags = {"vph": "HR-R", "insmode": "LCB"}
    traces = calibration(TracesMap, vph="LR-U", insmode="LCB")
    recipe = UncheckedRecipe()
    recipe.validate_input(recipe.create_input(obresult=ob, master_traces=traces))


def test_checks_of_input():
    """The check is run by the validation of the input, it does not need the recipe"""
    rinput = recipe_input(calibration(TracesMap, vph="LR-U", insmode="LCB"))
    assert isinstance(rinput, CheckedRecipeInput)
    with pytest.raises(
        ValidationError, match="TagsChecker: requirement master_traces: the value of master_traces has vph=LR-U"
    ):
        rinput.validate()
