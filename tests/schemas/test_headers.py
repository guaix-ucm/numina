"""Validation of the headers of FITS images with JSON schemas"""

import astropy.io.fits as fits
import numpy
import pytest

import numina.schemas as schemas
from numina.exceptions import ValidationError
from numina.types.dataframe import DataFrame
from numina.types.frame import DataFrameType

# The headers of images of one or two HDUs, the primary with EXPTIME
# and READMODE, the second, if present, with EXTNAME FIBERS
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "array",
    "prefixItems": [{"$ref": "#/$defs/primary_hdu"}, {"$ref": "#/$defs/fibers_hdu"}],
    "minItems": 1,
    "maxItems": 2,
    "$defs": {
        "primary_hdu": {
            "type": "object",
            "properties": {"values": {"$ref": "#/$defs/primary_values"}},
            "required": ["values"],
        },
        "primary_values": {
            "type": "object",
            "properties": {
                "EXPTIME": {"type": "number", "minimum": 0},
                "READMODE": {"enum": ["NORMAL", "MIRROR"]},
            },
            "required": ["EXPTIME"],
        },
        "fibers_hdu": {
            "type": "object",
            "properties": {"values": {"properties": {"EXTNAME": {"const": "FIBERS"}}}},
        },
    },
}


def image(fibers=None, **keywords):
    hdu = fits.PrimaryHDU(numpy.zeros((2, 2)))
    for key, value in keywords.items():
        hdu.header[key] = value
    hdus = [hdu]
    if fibers is not None:
        hdus.append(fits.ImageHDU(name=fibers))
    return fits.HDUList(hdus)


def test_convert_header():
    header = fits.Header()
    header["EXPTIME"] = (10.0, "exposure time")
    header["READMODE"] = "NORMAL"
    converted = schemas.convert_header(header)
    assert converted["values"] == {"EXPTIME": 10.0, "READMODE": "NORMAL"}
    assert converted["comments"] == {"EXPTIME": "exposure time", "READMODE": ""}
    assert converted["ordering"] == ["EXPTIME", "READMODE"]


def test_convert_headers():
    converted = schemas.convert_headers(image(fibers="FIBERS", EXPTIME=1.0))
    assert len(converted) == 2
    assert converted[0]["values"]["EXPTIME"] == 1.0
    assert converted[1]["values"]["EXTNAME"] == "FIBERS"


@pytest.mark.parametrize("schema", [SCHEMA, schemas.schema_validator(SCHEMA)])
def test_validate_headers(schema):
    schemas.validate_headers(image(fibers="FIBERS", EXPTIME=1.0, READMODE="MIRROR"), schema)


@pytest.mark.parametrize(
    "img, msg",
    [
        (image(), "HDU 0: 'EXPTIME' is a required property"),
        (image(EXPTIME=-1.0), "HDU 0, keyword EXPTIME: -1.0 is less than the minimum of 0"),
        (image(EXPTIME=1.0, READMODE="FAST"), "HDU 0, keyword READMODE: 'FAST' is not one of"),
        (image(fibers="OTHER", EXPTIME=1.0), "HDU 1, keyword EXTNAME: 'FIBERS' was expected"),
    ],
)
def test_validate_headers_invalid(img, msg):
    with pytest.raises(schemas.SchemaValidationError, match=msg) as excinfo:
        schemas.validate_headers(img, SCHEMA)
    # The command line shows ValidationError without a traceback
    assert isinstance(excinfo.value, ValidationError)


def test_validate_headers_number():
    hdul = image(fibers="FIBERS", EXPTIME=1.0)
    hdul.append(fits.ImageHDU())
    with pytest.raises(schemas.SchemaValidationError, match="is too long"):
        schemas.validate_headers(hdul, SCHEMA)


def test_validate_schema_values_of_hdu():
    validator = schemas.definition_validator(schemas.schema_validator(SCHEMA), "primary_values")
    schemas.validate_schema(validator, {"EXPTIME": 1.0}, hdu=0)
    with pytest.raises(schemas.SchemaValidationError, match="HDU 0, keyword EXPTIME: 'a' is not of type 'number'"):
        schemas.validate_schema(validator, {"EXPTIME": "a"}, hdu=0)


def test_validate_schema_path():
    validator = schemas.schema_validator({"properties": {"a": {"items": {"type": "integer"}}}})
    with pytest.raises(schemas.SchemaValidationError, match="^a/1: 'b' is not of type 'integer'"):
        schemas.validate_schema(validator, {"a": [1, "b"]})
    with pytest.raises(schemas.SchemaValidationError, match="^1 is not of type 'object'"):
        schemas.validate_schema(schemas.schema_validator({"type": "object"}), 1)


def test_definition_validator_draft7():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {"positive": {"type": "number", "minimum": 0}, "value": {"$ref": "#/definitions/positive"}},
    }
    validator = schemas.definition_validator(schemas.schema_validator(schema), "value")
    schemas.validate_schema(validator, 1)
    with pytest.raises(schemas.SchemaValidationError, match="less than the minimum"):
        schemas.validate_schema(validator, -1)


def test_relevant_error_one_of():
    # best_match does not choose between alternatives with errors equally
    # relevant, the alternative with fewer errors is reported
    schema = {
        "type": "array",
        "prefixItems": [
            {"properties": {"values": {"oneOf": [{"required": ["OBSMODE", "OBSTYPE"]}, {"required": ["IMAGETYP"]}]}}}
        ],
    }
    with pytest.raises(schemas.SchemaValidationError, match="^HDU 0: 'IMAGETYP' is a required property"):
        schemas.validate_headers(image(), schema)


def test_relevant_error_deepest():
    # The alternative that fails deepest in the instance is reported
    validator = schemas.schema_validator(
        {"oneOf": [{"required": ["a", "b"]}, {"properties": {"a": {"properties": {"c": {"type": "string"}}}}}]}
    )
    with pytest.raises(schemas.SchemaValidationError, match="^a/c: 1 is not of type 'string'"):
        schemas.validate_schema(validator, {"a": {"c": 1}})


class HeaderType(DataFrameType):
    header_schema = SCHEMA


def test_dataframetype_header_schema():
    HeaderType().validate(DataFrame(frame=image(EXPTIME=1.0)))
    with pytest.raises(ValidationError, match="'EXPTIME' is a required property"):
        HeaderType().validate(DataFrame(frame=image()))


def test_dataframetype_without_header_schema():
    assert DataFrameType().validate(DataFrame(frame=image()))
