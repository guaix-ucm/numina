#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""JSON schemas of the files read by numina

The schemas are stored in this package as ``<name>-schema.json``::

    from numina.schemas import validate

    validate(data, "control", source="control.yaml")

The headers of FITS images are validated with schemas of the DRPs,
see :func:`validate_headers`.
"""

import functools
import importlib.resources
import json

import jsonschema
import jsonschema.exceptions
import jsonschema.validators

from numina.exceptions import ValidationError


class SchemaValidationError(ValidationError, ValueError):
    """The contents of a file do not follow its schema

    It is a :class:`~numina.exceptions.ValidationError`, so the command
    line shows the message without a traceback, and a ``ValueError``,
    as in previous versions.
    """


@functools.cache
def load_schema(name):
    """Load the schema ``<name>-schema.json`` of this package"""
    resource = importlib.resources.files(__name__).joinpath(f"{name}-schema.json")
    with resource.open() as fd:
        return json.load(fd)


#: Names of the JSON types of Python values
_JSON_TYPES = {dict: "object", list: "array", str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(value):
    """Return the name of the JSON type of `value`, or of its Python type if it has none."""
    if value is None:
        return "null"
    return _JSON_TYPES.get(type(value), type(value).__name__)


def _message(error):
    """Message of the error, without the (possibly long) invalid value"""
    if error.validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            expected = " or ".join(expected)
        return f"expected {expected}, found {_json_type(error.instance)}"
    return error.message


def _location(error):
    """Return the path of the error in the instance, as ``key -> key``, or ``top level``."""
    path = [str(part) for part in error.absolute_path]
    return " -> ".join(path) if path else "top level"


def validate(obj, name, source=None):
    """Validate obj with the schema ``<name>-schema.json``

    The validator is selected by the ``$schema`` of the schema.

    Raises
    ------
    SchemaValidationError
        If obj is not valid. The message includes `source` (a filename,
        for example), the location of the error in obj and its cause.
    """
    schema = load_schema(name)
    validator_class = jsonschema.validators.validator_for(schema)
    validator = validator_class(schema)
    error = jsonschema.exceptions.best_match(validator.iter_errors(obj))
    if error is not None:
        prefix = f"{source}: " if source else ""
        msg = f"{prefix}invalid {name} file, at {_location(error)}: {_message(error)}"
        raise SchemaValidationError(msg) from error


def convert_header(header):
    """Convert a FITS header to a dictionary that can be validated with a JSON schema.

    The dictionary has the keys "values" and "comments", with the values
    and comments of each keyword, and "ordering", the list of keywords.
    """
    hdu_v = {}
    hdu_c = {}
    hdu_o = []
    hdu_repr = {"values": hdu_v, "comments": hdu_c, "ordering": hdu_o}

    for card in header.cards:
        key = card.keyword
        hdu_v[key] = card.value
        hdu_c[key] = card.comment
        hdu_o.append(key)

    return hdu_repr


def convert_headers(hdulist):
    """Convert the headers of `hdulist` to a list of dictionaries, see :func:`convert_header`"""
    return [convert_header(hdu.header) for hdu in hdulist]


def _leaf_errors(error):
    """Yield the errors without sub-errors (``context``) under `error`."""
    if error.context:
        for sub_error in error.context:
            yield from _leaf_errors(sub_error)
    else:
        yield error


def relevant_error(error):
    """The most relevant leaf error of a schema error

    In an error of anyOf or oneOf, the relevant alternative is the
    one that fails deepest in the instance, with fewer errors.
    """
    if not error.context:
        return error
    alternatives = {}
    for sub_error in error.context:
        alternatives.setdefault(sub_error.relative_schema_path[0], []).extend(_leaf_errors(sub_error))
    leaves = min(alternatives.values(), key=lambda errors: (-max(len(e.absolute_path) for e in errors), len(errors)))
    return max(leaves, key=lambda e: len(e.absolute_path))


def validate_schema(validator, instance, hdu=None):
    """Validate `instance` with `validator`, raise an error with a short message

    The message names the HDU and the keyword, for example
    "HDU 0, keyword EXPTIME: ...", using the most relevant error,
    see :func:`relevant_error`. `instance` are the converted headers
    of an image, see :func:`convert_headers`, or the values of the
    header of the HDU `hdu`.

    Raises
    ------
    SchemaValidationError
        If `instance` is not valid.
    """
    error = jsonschema.exceptions.best_match(validator.iter_errors(instance))
    if error is None:
        return
    error = relevant_error(error)
    path = list(error.absolute_path)
    if hdu is not None:
        path = [hdu, "values"] + path
    if len(path) >= 3 and path[1] == "values":
        where = f"HDU {path[0]}, keyword {path[2]}: "
    elif len(path) == 2 and path[1] == "values":
        where = f"HDU {path[0]}: "
    elif path:
        where = "/".join(str(part) for part in path) + ": "
    else:
        where = ""
    raise SchemaValidationError(where + error.message) from error


def schema_validator(schema):
    """The validator of `schema`, selected by its ``$schema``"""
    return jsonschema.validators.validator_for(schema)(schema)


def definition_validator(validator, name):
    """Validator of the definition `name` of the schema of `validator`

    The other definitions are included, so that the references
    of the definition can be resolved.
    """
    key = "$defs" if "$defs" in validator.schema else "definitions"
    schema = {"$ref": f"#/{key}/{name}", key: validator.schema[key]}
    return type(validator)(schema)


def validate_headers(hdulist, schema):
    """Validate the headers of `hdulist` with a JSON schema

    The schema describes the list of headers converted with
    :func:`convert_headers`, for example::

        {
            "type": "array",
            "prefixItems": [
                {"properties": {"values": {"required": ["EXPTIME"]}}}
            ]
        }

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList
    schema : dict or jsonschema validator

    Raises
    ------
    SchemaValidationError
        If the headers are not valid, see :func:`validate_schema`.
    """
    validator = schema if hasattr(schema, "iter_errors") else schema_validator(schema)
    validate_schema(validator, convert_headers(hdulist))
