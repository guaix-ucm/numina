"""ResultOf with node='last', that is not implemented"""

import pytest

import numina.core
from numina.core.oresult import ObservationResult
from numina.core.query import ResultOf
from numina.exceptions import NoResultFound

from ..dictdal import HybridDAL

QUERY_LAST = ResultOf("MODE.value", node="last")


@pytest.fixture
def dal():
    """The DAL created by numina run"""
    return HybridDAL(None, "", [], {})


def test_search_result_relative_last(dal):
    with pytest.raises(NoResultFound, match="node 'last' is not implemented"):
        dal.search_result_relative("value", None, ObservationResult(), result_desc=QUERY_LAST)


def test_requirement_query_last(dal):
    """The query raises NoResultFound, as any other result not found"""
    req = numina.core.Requirement(int, "A value", destination="value", query_opts=QUERY_LAST)

    with pytest.raises(NoResultFound):
        req.query(dal, ObservationResult())


class LastRecipe(numina.core.BaseRecipe):
    value = numina.core.Requirement(int, "A value", optional=True, default=3, query_opts=QUERY_LAST)

    def run(self, rinput):
        return self.create_result()


def test_build_recipe_input_last_uses_default(dal):
    """An optional requirement that queries node='last' takes its default"""
    recipe = LastRecipe()

    rinput = recipe.build_recipe_input(ObservationResult(), dal)

    assert rinput.value == 3
