"""The tags of the products of the input of a recipe, checked with TagsChecker"""

import logging

import pytest

from numina.core import BaseRecipe, ObservationResult, Parameter, Requirement
from numina.core.recipeinout import RecipeInput, TagsChecker
from numina.core.requirements import ObservationResultRequirement
from numina.exceptions import ValidationError
from numina.keydef import QueryAttribute
from numina.types.datatype import ListOfType, PlainPythonType
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


class ListedRecipe(BaseRecipe):
    """A recipe with lists, of values and of products"""

    RecipeInput = CheckedRecipeInput
    obresult = ObservationResultRequirement()
    nlines = Parameter([5], "A list of values", as_list=True)
    knots = Requirement(
        MultiType(PlainPythonType(ref=3), ListOfType(PlainPythonType(ref=0.0))),
        "A value or a list of values",
        default=3,
        optional=True,
    )
    all_traces = Requirement(ListOfType(TracesMap), "A list of products", optional=True)


@pytest.fixture(autouse=True)
def drp_test1(drpmocker):
    """The DRP of TEST1, used to validate the observation result"""
    drpmocker.add_drp("TEST1", make_drp({"traced": TracedRecipe, "listed": ListedRecipe}))


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


def listed_input(**kwargs):
    ob = ObservationResult(instrument="TEST1", mode="listed")
    ob.tags = {"vph": "HR-R", "insmode": "LCB"}
    return ListedRecipe().create_input(obresult=ob, **kwargs)


@pytest.mark.parametrize("knots", [3, [1.0, 2.0, 3.0]])
def test_lists_of_values(knots):
    """Lists of values are not checked (#279)"""
    ListedRecipe().validate_input(listed_input(nlines=[5, 10], knots=knots))


def test_list_of_products():
    """The tags of each element of a list of products are checked"""
    traces = [calibration(TracesMap, vph="HR-R", insmode="LCB"), calibration(TracesMap, vph="HR-R", insmode="LCB")]
    ListedRecipe().validate_input(listed_input(all_traces=traces))

    traces[1] = calibration(TracesMap, vph="LR-U", insmode="LCB")
    msg = "requirement all_traces: the element 1 of all_traces has vph=LR-U, the observing block has vph=HR-R"
    with pytest.raises(ValidationError, match=msg):
        ListedRecipe().validate_input(listed_input(all_traces=traces))


def test_found_tags_complete_the_value():
    """The tags with which the value was found are used for those it does not have"""
    rinput = recipe_input(calibration(TracesMap, vph="HR-R"))
    rinput.found_tags()["master_traces"] = {"vph": "LR-U", "insmode": "LCB"}
    TracedRecipe().validate_input(rinput)

    rinput.found_tags()["master_traces"] = {"insmode": "MOS"}
    with pytest.raises(ValidationError, match="the value of master_traces has insmode=MOS"):
        TracedRecipe().validate_input(rinput)


def test_found_tags_of_list():
    """The tags of each element of a list searched element by element"""
    traces = [calibration(TracesMap), calibration(TracesMap)]
    rinput = listed_input(all_traces=traces)
    rinput.found_tags()["all_traces"] = [{"vph": "HR-R", "insmode": "LCB"}, {"vph": "LR-U", "insmode": "LCB"}]
    msg = "requirement all_traces: the element 1 of all_traces has vph=LR-U, the observing block has vph=HR-R"
    with pytest.raises(ValidationError, match=msg):
        ListedRecipe().validate_input(rinput)


def test_unknown_tags(caplog):
    """A product without tags is not checked, with a warning"""
    rinput = recipe_input(calibration(TracesMap))
    with caplog.at_level(logging.WARNING):
        TracedRecipe().validate_input(rinput)
    assert "requirement master_traces: the tags of the value of master_traces are unknown" in caplog.text


def test_no_warning_without_tags_in_the_query(caplog):
    """A type without tags in its query does not need the tags of its value"""

    class Untagged(BaseStructuredCalibration):
        pass

    class UntaggedRecipe(BaseRecipe):
        RecipeInput = CheckedRecipeInput
        obresult = ObservationResultRequirement()
        calib = Requirement(Untagged, "A product without tags")

    ob = ObservationResult(instrument="TEST1", mode="traced")
    ob.tags = {"vph": "HR-R"}
    recipe = UntaggedRecipe()
    with caplog.at_level(logging.WARNING):
        recipe.validate_input(recipe.create_input(obresult=ob, calib=calibration(Untagged)))
    assert "unknown" not in caplog.text
