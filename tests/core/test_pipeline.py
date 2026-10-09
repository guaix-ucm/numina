import pytest

import numina.core.pipelineload as loader


@pytest.fixture(scope="module")
def drptest():
    return loader.drp_load("numina.drps.tests", "drptest1.yaml")


def test_mode_search(drptest):

    ll = drptest.search_mode_provides("MasterBias")

    assert ll.name == "MasterBias"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_bias"

    ll = drptest.search_mode_provides("MasterDark")

    assert ll.name == "MasterDark"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_dark"


def test_mode_query(drptest):

    ll = drptest.query_provides("MasterBias")

    assert ll.name == "MasterBias"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_bias"

    with pytest.raises(ValueError):
        drptest.query_provides("MasterDark")

    ll = drptest.query_provides("MasterDark", search=True)

    assert ll.name == "MasterDark"
    assert ll.mode == drptest.modes[ll.mode].key
    assert ll.field == "master_dark"


DRP_PIPELINES = """
name: TEST1
version: "2.5"
configurations:
  path: numina.drps.tests.configs
  values: []
modes:
  - key: dark
    name: Dark
pipelines:
  default:
    recipes:
      dark: numina.tests.recipes.DarkRecipe
    version: 1
  fast:
    recipes:
      dark: numina.tests.recipes.DarkRecipe
    version: 1
"""


@pytest.mark.parametrize("pipeline", ["default", "fast"])
def test_recipe_drp_headers(pipeline):
    drp = loader.drp_load_data("numina", DRP_PIPELINES)
    recipe = drp.get_recipe_object("dark", pipeline_name=pipeline)

    assert recipe.instrument == "TEST1"
    assert recipe.pipeline == pipeline
    assert recipe.drp_version == "2.5"

    hdr = recipe.set_base_headers({})
    assert hdr["NUMDRP"] == ("TEST1", "Numina DRP name")
    assert hdr["NUMDRPV"] == ("2.5", "Numina DRP version")
    assert hdr["NUMPIPE"] == (pipeline, "Numina DRP pipeline")


def test_recipe_drp_version_of_package():
    """Without version in the DRP, the version of the package"""
    import numina

    drp = loader.drp_load_data("numina", DRP_PIPELINES.replace('version: "2.5"\n', ""))
    recipe = drp.get_recipe_object("dark")

    assert recipe.drp_version == numina.__version__
    assert recipe.set_base_headers({})["NUMDRPV"][0] == numina.__version__
