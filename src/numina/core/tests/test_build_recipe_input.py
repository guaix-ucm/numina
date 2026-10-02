"""The observation result passed to build_recipe_input"""

from numina.core.oresult import ObservationResult, ObservingBlock
from numina.core.utils import OBSuccessRecipe


def test_obsres_is_used():
    """The recipe receives the same object, so the caller sees the changes"""
    recipe = OBSuccessRecipe()
    obsres = ObservationResult(instrument="TEST1", mode="mode1")

    rinput = recipe.build_recipe_input(obsres, None)

    assert rinput.obresult is obsres
    assert obsres.tags == {}


def test_oblock_is_converted():
    """A plain ObservingBlock is converted, sharing its attributes"""
    recipe = OBSuccessRecipe()
    oblock = ObservingBlock(instrument="TEST1", mode="mode1")

    rinput = recipe.build_recipe_input(oblock, None)

    assert isinstance(rinput.obresult, ObservationResult)
    assert rinput.obresult.__dict__ is oblock.__dict__
    assert oblock.tags == {}
