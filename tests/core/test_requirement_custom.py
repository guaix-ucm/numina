"""The search of the requirements: customization and the observation result"""

import pytest

import numina.core
import numina.dal.stored
import numina.exceptions
import numina.types.datatype as dt
from numina.types.structured import BaseStructuredCalibration
from numina.core.dataholders import Requirement
from numina.core.recipes import BaseRecipe
from numina.core.requirements import ObservationResultRequirement


class One(BaseStructuredCalibration):
    pass


class RecordDal:
    """A DAL that records the observation results that it receives"""

    def __init__(self, original):
        self.original = original
        self.calls = []

    def search_product(self, name, stype, obsres, options=None):
        # the original observation result is not modified during the search
        self.calls.append((obsres is self.original, obsres.tags, self.original.tags))
        return numina.dal.stored.StoredProduct(id=1, content=obsres.tags["a"], tags={})

    def search_parameter(self, name, stype, obsres, options=None):
        raise numina.exceptions.NoResultFound(name)


def test_obsres_not_modified_list():
    tags = [{"a": 1}, {"a": 2}]
    obsres = numina.core.ObservationResult()
    obsres.tags = tags
    dal = RecordDal(obsres)

    req = Requirement(dt.ListOfType(One), description="list", destination="req")
    assert req.query(dal, obsres) == [1, 2]

    # each query receives a copy with its tags
    assert dal.calls == [(False, {"a": 1}, tags), (False, {"a": 2}, tags)]
    assert obsres.tags is tags


def test_obsres_not_modified_scalar():
    tags = [{"a": 1}, {"a": 2}]
    obsres = numina.core.ObservationResult()
    obsres.tags = tags
    dal = RecordDal(obsres)

    req = Requirement(One, description="scalar", destination="req")
    assert req.query(dal, obsres) == 1

    # the first set of tags is used
    assert dal.calls == [(False, {"a": 1}, tags)]
    assert obsres.tags is tags


def test_obsres_not_copied_scalar_tags():
    obsres = numina.core.ObservationResult()
    obsres.tags = {"a": 3}
    dal = RecordDal(obsres)

    req = Requirement(One, description="scalar", destination="req")
    assert req.query(dal, obsres) == 3

    assert dal.calls == [(True, {"a": 3}, {"a": 3})]


class LabelRequirement(Requirement):
    """Searches the value in the labels of the observation result"""

    def query_on_ob(self, ob):
        try:
            return ob.labels[self.dest]
        except KeyError:
            raise numina.exceptions.NoResultFound(self.dest)


class DefaultProductRequirement(Requirement):
    """Returns a fixed value instead of querying the DAL"""

    def query_on_dal_base(self, next_type, dal, obsres, options=None):
        return 42


class ExplainRequirement(Requirement):
    """Raises an exception with a message when the value is not found"""

    def on_query_not_found(self, notfound):
        raise ValueError(f"'{self.dest}' not found, add it to the observing block")


def test_custom_query_on_ob():
    obsres = numina.core.ObservationResult()
    obsres.labels = {"req": 7}
    dal = RecordDal(obsres)

    req = LabelRequirement(dt.PlainPythonType(ref=1), description="label", destination="req")
    assert req.query(dal, obsres) == 7
    assert dal.calls == []


def test_custom_query_on_dal_base():
    obsres = numina.core.ObservationResult()
    obsres.tags = [{"a": 1}, {"a": 2}]
    dal = RecordDal(obsres)

    req = DefaultProductRequirement(dt.ListOfType(One), description="list", destination="req")
    assert req.query(dal, obsres) == [42, 42]
    assert dal.calls == []


def test_custom_on_query_not_found():

    class RecipeTest(BaseRecipe):
        obresult = ObservationResultRequirement()
        param = ExplainRequirement(dt.PlainPythonType(ref=1), description="param")

    recipe = RecipeTest()
    obsres = numina.core.ObservationResult(instrument="TEST1", mode="mode1")
    dal = RecordDal(obsres)

    with pytest.raises(ValueError, match="'param' not found"):
        recipe.build_recipe_input(obsres, dal)
