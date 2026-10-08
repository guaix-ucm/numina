"""Schema of the control file"""

import jsonschema
import pytest

from numina.schemas import SchemaValidationError, load_schema, validate

PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"

VALID = {
    "version": 1,
    "rootdir": "calibs",
    "products": {
        "EMIR": {
            PROFILE: [
                {"id": 0, "type": "MasterBadPixelMask", "tags": {}, "content": "mask_bpm.fits"},
                {"id": 1, "type": "MasterDark", "tags": {"readmode": "RAMP"}},
            ]
        }
    },
    "requirements": {
        "MEGARA": {
            PROFILE: {
                "default": {
                    "MegaraArcCalibration": [
                        {"name": "nlines", "tags": {"vph": "LR-U", "speclamp": "ThAr"}, "content": [25, 25]}
                    ]
                }
            }
        }
    },
}


def test_schema_is_valid():
    schema = load_schema("control")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("obj", [VALID, {}, {"version": 1}, {"parameters": VALID["requirements"]}])
def test_valid(obj):
    validate(obj, "control")


@pytest.mark.parametrize(
    "obj, msg",
    [
        (
            {"products": {"EMIR": [{"id": 1, "type": "MasterDark", "tags": {}}]}},
            "at products -> EMIR: expected object, found array",
        ),
        (
            {"products": {"EMIR": {PROFILE: [{"type": "MasterDark", "tags": {}}]}}},
            f"at products -> EMIR -> {PROFILE} -> 0: 'id' is a required property",
        ),
        ({"requirement": {}}, "at top level: Unevaluated properties are not allowed ('requirement' was unexpected)"),
        ({"version": 2}, "at version: 1 was expected"),
        (
            {"requirements": {"MEGARA": {PROFILE: {"default": {"bias": [{"name": "x", "content": 1}]}}}}},
            f"at requirements -> MEGARA -> {PROFILE} -> default -> bias -> 0: 'tags' is a required property",
        ),
        (
            {"products": {"EMIR": {PROFILE: [{"id": 1, "type": "MasterDark", "tags": {}, "path": "x"}]}}},
            "('path' was unexpected)",
        ),
        ([], "at top level: expected object, found array"),
    ],
)
def test_invalid(obj, msg):
    with pytest.raises(SchemaValidationError, match=r"^control\.yaml: invalid control file, ") as excinfo:
        validate(obj, "control", source="control.yaml")
    assert msg in str(excinfo.value)
