#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Envelope of the self-describing calibration files

The envelope says what a file is, in which format, when it applies
and where it comes from. It has the same keys in all the containers,
and it is validated with the schema ``envelope-schema.json`` of
:mod:`numina.schemas`:

ECSV
    The ``meta`` of the YAML header of the table. Only the header is
    read.
FITS
    Keywords of the primary header, see :data:`FITS_KEYWORDS`. A FITS
    file has an envelope if it has the keyword ``NUMENV``. Only the
    primary header is read.
JSON
    An object ``meta`` in the top level, separated from ``data``. The
    whole file is parsed, but ``data`` is not interpreted.

A file without envelope is format 1.

A type declares the formats of its files that numina reads in the
class attribute ``__numina_formats__``, a mapping from the number of the
format to the name of the method that reads a file in that format.
:func:`load` checks the type and the format of the envelope and calls
that method.
"""

import builtins
import datetime
import gzip
import json
import logging
import os

from numina.exceptions import ValidationError
from numina.schemas import validate

_logger = logging.getLogger(__name__)

#: Version of the envelope written by this version of numina
ENVELOPE_VERSION = 1

#: Keywords of the primary header of a FITS file, and their keys in the envelope
FITS_KEYWORDS = {
    "NUMENV": "envelope",
    "NUMTYPE": "type",
    "NUMFMT": "format",
    "INSTRUME": "instrument",
    "UUID": "uuid",
    "INSCONF": "profile",
}

#: Format of the files without envelope
DEFAULT_FORMAT = 1


class FormatError(ValidationError, ValueError):
    """A file cannot be read: it has a different type, or a format that is not supported"""


def container(path):
    """The container of the file `path`: ``"ecsv"``, ``"fits"``, ``"json"`` or None

    The container is deduced from the first bytes of the file, gzip
    compressed FITS files included, not from the extension.
    """
    with builtins.open(path, "rb") as fd:
        start = fd.read(80)
    if start.startswith(b"\x1f\x8b"):
        with gzip.open(path, "rb") as fd:
            start = fd.read(80)
    if start.startswith(b"# %ECSV"):
        return "ecsv"
    if start.startswith(b"SIMPLE  ="):
        return "fits"
    if start.lstrip().startswith(b"{"):
        return "json"
    return None


def _isoformat(value):
    """Convert the dates of `value` (YAML loads them as dates) to ISO 8601 strings."""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _isoformat(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_isoformat(val) for val in value]
    return value


def read_ecsv_meta(path):
    """The ``meta`` of the ECSV file `path`, reading only its header"""
    from astropy.table.meta import get_header_from_yaml

    lines = []
    with builtins.open(path, encoding="utf-8") as fd:
        for line in fd:
            if not line.startswith("#"):
                break
            lines.append(line[2:] if line.startswith("# ") else line[1:])
    # The first line is "%ECSV VERSION"
    header = get_header_from_yaml(lines[1:])
    return _isoformat(dict(header.get("meta", {})))


def _read_ecsv(path):
    """The envelope of an ECSV file, or None if its meta has no ``envelope``"""
    meta = read_ecsv_meta(path)
    if "envelope" not in meta:
        return None
    return meta


def _read_fits(path):
    """The envelope of a FITS file, from its primary header, or None if it has no ``NUMENV``"""
    from astropy.io import fits

    header = fits.getheader(path, ext=0)
    if "NUMENV" not in header:
        return None
    envelope = {key: header[keyword] for keyword, key in FITS_KEYWORDS.items() if keyword in header}
    envelope["tags"] = {}
    return envelope


def _read_json(path):
    """The envelope of a JSON file, its object ``meta``, or None if it has none

    The whole file is parsed, JSON cannot be read in parts.
    """
    with builtins.open(path, encoding="utf-8") as fd:
        data = json.load(fd)
    meta = data.get("meta") if isinstance(data, dict) else None
    if not isinstance(meta, dict) or "envelope" not in meta:
        return None
    return meta


_READERS = {"ecsv": _read_ecsv, "fits": _read_fits, "json": _read_json}


def read_envelope(path):
    """The envelope of the file `path`, or None if the file has no envelope

    The envelope is validated with ``envelope-schema.json``.

    Raises
    ------
    SchemaValidationError
        If the envelope is not valid.
    """
    reader = _READERS.get(container(path))
    if reader is None:
        return None
    envelope = reader(path)
    if envelope is not None:
        validate(envelope, "envelope", source=os.fspath(path))
    return envelope


def _formats_owner(tipo):
    """The top level package of the class that declares the formats of `tipo`."""
    for cls in type(tipo).__mro__:
        if "__numina_formats__" in vars(cls):
            return cls.__module__.split(".")[0]
    return type(tipo).__module__.split(".")[0]


def _numbers(formats):
    """The numbers of `formats` as text, ``1, 2``."""
    return ", ".join(str(num) for num in sorted(formats))


def check_envelope(tipo, path, envelope):
    """The format of the file `path`, checking that the type `tipo` reads it

    The format of a file without envelope (`envelope` is None) is 1.

    The type of the envelope is checked only if `tipo` is a data
    product, the types that are not products (an ``ArrayType``, for
    example) read the files of any type.

    Raises
    ------
    FormatError
        If the type of the envelope is not the name of `tipo`, or
        `tipo` does not read the format of the envelope.
    """
    formats = tipo.__numina_formats__
    name = tipo.name()
    if envelope is None:
        fmt = DEFAULT_FORMAT
    else:
        fmt = envelope["format"]
        if tipo.isproduct() and envelope["type"] != name:
            raise FormatError(f"{os.fspath(path)}: the file is a {envelope['type']}, not a {name}")
    if fmt not in formats:
        if envelope is None:
            msg = f"{os.fspath(path)}: the file has no envelope, and {name} only reads formats {_numbers(formats)}"
        else:
            msg = (
                f"{os.fspath(path)}: {name} in format {fmt}, this version reads formats {_numbers(formats)}; "
                f"update {_formats_owner(tipo)}"
            )
        raise FormatError(msg)
    return fmt


def is_file(obj):
    """True if `obj` is the name of an existing file"""
    return isinstance(obj, (str, os.PathLike)) and os.path.isfile(obj)


def load(tipo, path):
    """Load the value of type `tipo` from the file `path`, with the reader of its format

    The envelope of the file is read and checked with
    :func:`check_envelope`. The value is read with the method of
    `tipo` named in ``__numina_formats__`` for the format.
    """
    envelope = read_envelope(path)
    fmt = check_envelope(tipo, path, envelope)
    _logger.debug("loading %s in format %d from %s", tipo.name(), fmt, path)
    reader = getattr(tipo, tipo.__numina_formats__[fmt])
    return reader(path)
