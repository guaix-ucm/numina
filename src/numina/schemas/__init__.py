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
"""

import functools
import importlib.resources
import json

import jsonschema
import jsonschema.exceptions
import jsonschema.validators


class SchemaValidationError(ValueError):
    """The contents of a file do not follow its schema"""


@functools.cache
def load_schema(name):
    """Load the schema ``<name>-schema.json`` of this package"""
    resource = importlib.resources.files(__name__).joinpath(f"{name}-schema.json")
    with resource.open() as fd:
        return json.load(fd)


#: Names of the JSON types of Python values
_JSON_TYPES = {dict: "object", list: "array", str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(value):
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
