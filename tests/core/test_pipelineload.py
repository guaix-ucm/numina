import pytest

from numina.core.pipelineload import drp_load_data, load_confs, load_mode


def test_load_confs_path():
    """The configurations are loaded from 'path'"""

    confs, modpath = load_confs("numina", {"path": "numina.testing.drps.configs"})

    assert modpath == "numina.testing.drps.configs"
    assert "instrument-test1.json" in confs
    assert confs["instrument-test1.json"]["name"] == "TEST1"


def test_load_confs_default_path():
    """Without 'path', the configurations are in package.instrument.configs"""

    with pytest.raises(ModuleNotFoundError) as excinfo:
        load_confs("numina", {})

    assert excinfo.value.name == "numina.instrument.configs"


@pytest.mark.parametrize("tagger", [None, ["KEY1"], "numina.core.taggers.extract_tags_from_obsres"])
def test_load_mode_ignores_tagger(tagger):
    """The per mode tagger of drp.yaml is ignored"""
    node = {"key": "bias", "name": "Bias", "summary": "", "description": "", "tagger": tagger}

    mode = load_mode(node)

    assert mode.key == "bias"
    assert not hasattr(mode, "tagger")


DRP_UNDEFINED_MODE = """
name: TEST1
configurations:
  path: numina.testing.drps.configs
  values: []
modes:
  - key: dark
    name: Dark
pipelines:
  default:
    recipes:
      dark: numina.testing.recipes.DarkRecipe
      other: numina.testing.recipes.DarkRecipe
    version: 1
"""


def test_recipe_of_undefined_mode():
    with pytest.warns(RuntimeWarning, match="pipeline 'default' has a recipe for the mode 'other'"):
        drp = drp_load_data("numina", DRP_UNDEFINED_MODE)
    # the rest of the DRP can be used
    assert drp.get_recipe_object("dark").mode.key == "dark"


def test_mode_defined_several_times():
    data = DRP_UNDEFINED_MODE.replace(
        "  - key: dark\n    name: Dark\n",
        "  - key: dark\n    name: Dark\n  - key: dark\n    name: Dark 2\n",
    ).replace("      other: numina.testing.recipes.DarkRecipe\n", "")
    with pytest.warns(RuntimeWarning, match="the mode 'dark' is defined several times"):
        drp = drp_load_data("numina", data)
    assert drp.modes["dark"].name == "Dark 2"
