

from ..fqn import fully_qualified_name


def test_fqn():

    import numina.core.recipes

    expected_name = "numina.core.recipes.BaseRecipe"
    fqn_name = fully_qualified_name(numina.core.recipes.BaseRecipe)

    assert fqn_name == expected_name
