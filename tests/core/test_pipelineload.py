import pytest

from numina.core.pipelineload import drp_load_data, load_confs, load_mode


def test_load_confs_path():
    """The configurations are loaded from 'path'"""

    confs, modpath = load_confs("numina", {"path": "tests.support.drps.configs"})

    assert modpath == "tests.support.drps.configs"
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
  path: tests.support.drps.configs
  values: []
modes:
  - key: dark
    name: Dark
pipelines:
  default:
    recipes:
      dark: tests.support.recipes.DarkRecipe
      other: tests.support.recipes.DarkRecipe
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
    ).replace("      other: tests.support.recipes.DarkRecipe\n", "")
    with pytest.warns(RuntimeWarning, match="the mode 'dark' is defined several times"):
        drp = drp_load_data("numina", data)
    assert drp.modes["dark"].name == "Dark 2"


def test_modes_have_instrument():
    """The observing modes know the name of their instrument"""
    drp = drp_load_data("numina", DRP_TEST1_MODES)

    assert drp.modes
    for mode in drp.modes.values():
        assert mode.instrument == "TEST1"
    assert repr(drp.modes["bias"]) == "ObservingMode(name=Bias, instrument=TEST1)"


def test_load_mode_without_instrument():
    node = {"key": "bias", "name": "Bias", "summary": "", "description": ""}
    assert load_mode(node).instrument == ""
    assert load_mode(node, instrument="TEST1").instrument == "TEST1"


def test_load_modes_dict():
    node = {"bias": {"key": "bias", "name": "Bias", "summary": "", "description": ""}}
    from numina.core.pipelineload import load_modes

    modes = load_modes(node, instrument="TEST1")
    assert modes["bias"].key == "bias"
    assert modes["bias"].instrument == "TEST1"


def test_observing_mode_validate_deprecated():
    from numina.core.pipeline import ObservingMode

    with pytest.warns(DeprecationWarning, match="ObservingMode.validate"):
        assert ObservingMode().validate(None)


DRP_TEST1_MODES = """
name: TEST1
configurations:
  path: tests.support.drps.configs
modes:
  - key: bias
    name: Bias
    summary: Bias mode
    description: Bias mode
pipelines:
  default:
    version: 1
    recipes:
      bias: numina.core.utils.AlwaysSuccessRecipe
"""


DRP_CALIBRATIONS = """
name: TEST1
configurations:
  path: tests.support.drps.configs
calibrations:
- name: cal-1.0
  url: https://example.org/cal-1.0.zip
  known_hash: {known_hash}
modes: []
pipelines:
  default:
    version: 1
    recipes: {{}}
"""


def test_load_calibrations():
    drp = drp_load_data("numina", DRP_CALIBRATIONS.format(known_hash="md5:0123abcd"))
    assert drp.calibrations == [
        {"name": "cal-1.0", "url": "https://example.org/cal-1.0.zip", "known_hash": "md5:0123abcd"}
    ]
    # without the key, no datasets
    assert drp_load_data("numina", DRP_TEST1_MODES).calibrations == []


def test_load_calibrations_invalid_hash():
    from numina.schemas import SchemaValidationError

    with pytest.raises(SchemaValidationError, match="calibrations"):
        drp_load_data("numina", DRP_CALIBRATIONS.format(known_hash="crc:0"))
