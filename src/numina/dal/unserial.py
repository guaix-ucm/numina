#
# Copyright 2019-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Load files stored in disk based on extension"""

import os
import logging

from numina.util.objimport import import_object

_logger = logging.getLogger(__name__)


# FIXME: this is already implemented, elsewhere
# There should be one-- and preferably only one --obvious way to do it.


def is_fits(filename, **kwargs):
    """Check whether `filename` has the extension ``.fits``."""
    return filename.endswith(".fits")


def read_fits_later(filename):
    """Return a ``DataFrame`` that refers to `filename`, without opening the file."""
    import numina.types.dataframe as df

    return df.DataFrame(filename=os.path.abspath(filename))


def is_json(filename, **kwargs):
    """Check whether `filename` has the extension ``.json``."""
    return filename.endswith(".json")


def read_json(filename):
    """Read the JSON file `filename` and build the object it describes, see ``read_structured``."""
    import json

    with open(filename) as fd:
        base = json.load(fd)
    return read_structured(base)


def is_yaml(filename, **kwargs):
    """Check whether `filename` has the extension ``.yaml``."""
    return filename.endswith(".yaml")


def read_yaml(filename):
    """Read the YAML file `filename` and build the object it describes, see ``read_structured``."""
    import yaml

    with open(filename) as fd:
        base = yaml.safe_load(fd)
    return read_structured(base)


def read_structured(data):
    """Build the object described by `data`, if it has the key ``type_fqn``.

    The object, of the class named by ``type_fqn``, is created with
    ``__setstate__(data)``. Without ``type_fqn``, `data` is returned unchanged.
    """
    if "type_fqn" in data:
        type_fqn = data["type_fqn"]
        cls = import_object(type_fqn)
        obj = cls.__new__(cls)
        obj.__setstate__(data)
        return obj
    return data


def unserial(value):
    """Load `value` if it is the name of a FITS, JSON or YAML file, otherwise return it unchanged."""
    checkers = [(is_fits, read_fits_later), (is_json, read_json), (is_yaml, read_yaml)]
    if isinstance(value, str):
        for check_type, conv in checkers:
            if check_type(value):
                return conv(value)
        else:
            return value
    else:
        return value
