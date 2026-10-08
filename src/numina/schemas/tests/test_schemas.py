"""Schemas of the DRP, component and observing block files"""

import json

import jsonschema
import pytest
import yaml

from numina.schemas import SchemaValidationError, load_schema, validate

UUID = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"


@pytest.mark.parametrize("name", ["control", "drp", "component", "oblock"])
def test_schema_is_valid(name):
    schema = load_schema(name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == f"https://guaix.fis.ucm.es/numina/{name}-schema.json"
    jsonschema.Draft202012Validator.check_schema(schema)


DRP = {
    "name": "TEST1",
    "configurations": {"path": "numina.drps.tests.configs"},
    "modes": [{"key": "bias", "name": "Bias", "summary": "Bias", "description": "Bias"}],
    "pipelines": {
        "default": {
            "version": 1,
            "recipes": {"bias": "numina.tests.recipes.BiasRecipe", "fail": {"class": "x.Y", "args": [1]}},
            "products": {"MasterBias": "numina.tests.recipes.MasterBias"},
            "provides": [{"name": "MasterBias", "mode": "bias", "field": "master_bias"}],
        }
    },
}


def with_changes(obj, path, value):
    """A copy of obj, with the value at path changed (None removes it)"""
    obj = json.loads(json.dumps(obj))
    node = obj
    for key in path[:-1]:
        node = node[key]
    if value is None:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return obj


def test_drp_valid():
    validate(DRP, "drp")


@pytest.mark.parametrize(
    "path, value, msg",
    [
        (["modes"], None, "at top level: 'modes' is a required property"),
        (["pipelines", "default"], None, "at pipelines: 'default' is a required property"),
        (["modes", 0, "key"], None, "at modes -> 0: 'key' is a required property"),
        (["modes", 0, "nmae"], "x", "('nmae' was unexpected)"),
        (["pipelines", "default", "recipes", "fail", "keys"], {}, "at pipelines -> default -> recipes -> fail"),
        (["configurations", "default"], "not-a-uuid", "at configurations -> default"),
    ],
)
def test_drp_invalid(path, value, msg):
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(with_changes(DRP, path, value), "drp", source="drp.yaml")
    assert msg in str(excinfo.value)


COMPONENT = {
    "name": "TEST1",
    "type": "instrument",
    "description": "An instrument",
    "uuid": UUID,
    "date_start": "2016-06-01T12:00:00.0",
    "date_end": None,
    "class": "numina.instrument.generic.InstrumentGeneric",
    "properties": [
        {"id": "const", "depends": [], "values": 1},
        {"id": "mode", "one_of": ["A", "B"], "default": "B", "key": "MODE", "ext": 0},
        {"id": "pos", "limits": [-12, 12]},
        {"id": "insmode", "references": "pseudoslit.insmode"},
        {"id": "spaces", "name": "Box"},
        {"id": "other", "uuid": UUID},
    ],
    "components": [{"name": "wheel"}, {"id": "detector", "uuid": UUID}],
    "setup": [{"id": "values", "values": {"a": 1}}, {"id": "setup1", "name": "Setup"}],
}


def test_component_valid():
    validate(COMPONENT, "component")


@pytest.mark.parametrize(
    "path, value, msg",
    [
        (["configurations"], {}, "('configurations' was unexpected)"),
        (["uuid"], "x", "at uuid"),
        (["type"], "detector", "at type"),
        (["date_end"], None, "'date_end' is a required property"),
        (["properties", 0, "depends"], None, "at properties -> 0"),
        (["components", 0], {"id": "x"}, "at components -> 0"),
    ],
)
def test_component_invalid(path, value, msg):
    obj = with_changes(COMPONENT, path, value)
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(obj, "component", source="component.json")
    assert msg in str(excinfo.value)


OBLOCK = {
    "id": 1,
    "instrument": "TEST1",
    "mode": "bias",
    "frames": ["image1.fits"],
    "children": [2, "three"],
    "enabled": False,
    "requirements": {"nlines": [25, 25]},
}


@pytest.mark.parametrize(
    "obj", [OBLOCK, with_changes(OBLOCK, ["id"], "0_bias"), {"id": 1, "instrument": "I", "mode": "m"}]
)
def test_oblock_valid(obj):
    validate(obj, "oblock")


@pytest.mark.parametrize(
    "path, value, msg",
    [
        (["mode"], None, "at top level: 'mode' is a required property"),
        (["requirement"], {}, "('requirement' was unexpected)"),
        (["frames"], "image1.fits", "at frames: expected array, found string"),
        (["enabled"], "no", "at enabled: expected boolean, found string"),
    ],
)
def test_oblock_invalid(path, value, msg):
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(with_changes(OBLOCK, path, value), "oblock", source="obs.yaml")
    assert msg in str(excinfo.value)


def test_component_file_is_validated(tmp_path):
    from numina.instrument.collection import load_paths_store

    (tmp_path / "bad.json").write_text(json.dumps(with_changes(COMPONENT, ["configurations"], {})))
    with pytest.raises(SchemaValidationError, match="bad.json: invalid component file"):
        load_paths_store(file_paths=[tmp_path])


def test_oblock_file_is_validated(tmp_path):
    from numina.user.helpers import load_observations

    obfile = tmp_path / "obs.yaml"
    obfile.write_text(yaml.safe_dump_all([OBLOCK, with_changes(OBLOCK, ["mode"], None)]))
    with pytest.raises(SchemaValidationError, match=r"obs.yaml \(document 2\): invalid oblock file"):
        load_observations([str(obfile)])


def test_oblock_file_empty_documents(tmp_path):
    from numina.user.helpers import load_observations

    obfile = tmp_path / "obs.yaml"
    obfile.write_text(yaml.safe_dump(OBLOCK) + "---\n")
    _, loaded = load_observations([str(obfile)])
    assert [ob["id"] for ob in loaded] == [1]
