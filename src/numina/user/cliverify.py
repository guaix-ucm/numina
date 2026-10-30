#
# Copyright 2019-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, verify functionality.

``numina verify`` checks files with the checkers of the DRPs, through
:data:`numina.core.config.check`, the same used to validate the inputs
of ``numina run --validate``. It can check:

* FITS images and JSON files, with the type deduced by the DRP of their
  instrument, or as the raw image of an observing mode (``--mode``);
* the raw images of observation results (``--ob``), as the raw image of
  their observing mode, and then with the validator of the mode.

Each file is reported as valid, invalid or not checked, and the exit
status is 1 if some file is invalid.
"""

import enum
import json
import logging
import os
import sys

import yaml

from numina.exceptions import ValidationError

_logger = logging.getLogger(__name__)


class Status(enum.Enum):
    """Result of the verification of a file."""

    VALID = "OK"
    INVALID = "INVALID"
    NOT_CHECKED = "NOT CHECKED"


def register(subparsers, config):
    """Add the ``verify`` command to the parser."""
    parser_verify = subparsers.add_parser(
        "verify",
        help="verify images and observation results",
        description=(
            "Verify files with the checks of the DRP of their instrument. "
            "The exit status is 1 if some file is invalid."
        ),
    )
    parser_verify.set_defaults(command=verify)
    group = parser_verify.add_mutually_exclusive_group()
    group.add_argument(
        "--ob",
        action="store_true",
        help="the files are observation results, verify their raw images and the observing mode",
    )
    group.add_argument(
        "--mode", help="verify the images as the raw images of this observing mode (MODE or INSTRUMENT.MODE)"
    )
    parser_verify.add_argument(
        "--datadir",
        help="directory of the raw images of the observation results, by default datadir in [tool.run]",
    )
    parser_verify.add_argument("files", nargs="+", help="files to verify")
    return parser_verify


def verify(args, extra_args, config):
    """Verify the files of the command line, print a line for each one"""
    import numina.drps

    drps = numina.drps.get_system_drps()

    if args.ob:
        datadir = args.datadir or config["tool.run"].get("datadir", "data")
        results = []
        for filename in args.files:
            results.extend(verify_obsresult_file(filename, datadir, drps))
    else:
        mode = None
        if args.mode is not None:
            try:
                mode = find_mode(drps, args.mode)
            except ValueError as error:
                print(f"numina verify: error: {error}", file=sys.stderr)
                return 2
        results = [verify_file(filename, mode=mode) for filename in args.files]

    for name, status, message in results:
        line = f"{status.value} {name}"
        if message:
            line += f": {message}"
        print(line)

    counts = {status: sum(1 for _, s, _ in results if s is status) for status in Status}
    print(
        f"{counts[Status.VALID]} valid, {counts[Status.INVALID]} invalid, " f"{counts[Status.NOT_CHECKED]} not checked"
    )
    return 1 if counts[Status.INVALID] else 0


def error_message(error):
    """The first line of the message of an error, with its type"""
    msg = str(error).splitlines()
    first = msg[0] if msg else repr(error)
    if isinstance(error, ValidationError):
        return first
    return f"{type(error).__name__}: {first}"


def check_object(instrument, obj, astype=None):
    """Check obj with the function of instrument, return (status, message)"""
    import numina.core.config as cfg

    if instrument not in cfg.check:
        return Status.NOT_CHECKED, f"no checks for instrument {instrument}"
    try:
        cfg.check(instrument, obj, astype=astype)
    except Exception as error:
        return Status.INVALID, error_message(error)
    return Status.VALID, ""


def find_mode(drps, mode_name, instrument=None):
    """The observing mode `mode_name` (MODE or INSTRUMENT.MODE)

    Without instrument, the mode is searched in all the DRPs.

    Raises
    ------
    ValueError
        If the mode is not found, or is found in several DRPs.
    """
    if instrument is None and "." in mode_name:
        instrument, mode_name = mode_name.split(".", 1)
    if instrument is None:
        candidates = [drp.modes[mode_name] for drp in drps.query_all().values() if mode_name in drp.modes]
        if not candidates:
            raise ValueError(f"no mode {mode_name} in any DRP")
        if len(candidates) > 1:
            raise ValueError(f"mode {mode_name} found in several DRPs, use INSTRUMENT.MODE")
        return candidates[0]
    try:
        drp = drps.query_by_name(instrument)
    except KeyError:
        raise ValueError(f"no DRP for instrument {instrument}")
    try:
        return drp.modes[mode_name]
    except KeyError:
        raise ValueError(f"no mode {mode_name} in the DRP of {instrument}")


def verify_file(filename, mode=None):
    """Verify a FITS or JSON file, return (filename, status, message)

    With `mode`, an observing mode, the image is checked as the raw image
    of the mode.
    """
    import astropy.io.fits as fits

    ext = os.path.splitext(filename[:-3] if filename.endswith(".gz") else filename)[1]
    try:
        if ext in [".fits", ".fit", ".fts"]:
            with fits.open(filename) as hdulist:
                instrument = hdulist[0].header.get("INSTRUME")
                if instrument is None:
                    return filename, Status.NOT_CHECKED, "no INSTRUME in the primary header"
                astype = None
                if mode is not None:
                    if instrument != mode.instrument:
                        msg = f"the instrument is {instrument}, the mode {mode.key} is of {mode.instrument}"
                        return filename, Status.INVALID, msg
                    astype = mode.rawimage
                status, message = check_object(instrument, hdulist, astype=astype)
        elif ext == ".json":
            with open(filename) as fd:
                obj = json.load(fd)
            instrument = obj.get("instrument") if isinstance(obj, dict) else None
            if instrument is None:
                return filename, Status.NOT_CHECKED, "no instrument field"
            status, message = check_object(instrument, obj)
        else:
            return filename, Status.NOT_CHECKED, "not a FITS or JSON file"
    except (OSError, ValueError) as error:
        return filename, Status.INVALID, error_message(error)
    return filename, status, message


def verify_obsresult_file(filename, datadir, drps):
    """Verify the observation results of a file, return a list of (name, status, message)

    Each raw image is checked as the raw image of the observing mode, and
    then the observation result with the validator of the mode, as with
    ``numina run --validate``.
    """
    try:
        with open(filename) as fd:
            docs = [doc for doc in yaml.safe_load_all(fd) if doc is not None]
    except (OSError, yaml.YAMLError) as error:
        return [(filename, Status.INVALID, error_message(error))]

    results = []
    for doc in docs:
        results.extend(verify_obsresult(doc, filename, datadir, drps))
    return results


def verify_obsresult(doc, filename, datadir, drps):
    """Verify an observation result read from filename"""
    from numina.core.oresult import ObservationResult
    from numina.schemas import SchemaValidationError, validate
    from numina.types.dataframe import DataFrame

    obname = f"{filename} (id {doc.get('id')})" if isinstance(doc, dict) else filename
    try:
        validate(doc, "oblock", source=filename)
        mode = find_mode(drps, doc["mode"], instrument=doc["instrument"])
    except (SchemaValidationError, ValueError) as error:
        # numina errors, with a readable message
        return [(obname, Status.INVALID, str(error))]

    obsres = ObservationResult(instrument=doc["instrument"], mode=doc["mode"])
    names = doc.get("frames", doc.get("images", []))
    obsres.frames = [DataFrame(filename=os.path.join(datadir, name)) for name in names]

    results = []
    for frame in obsres.frames:
        if mode.rawimage is None:
            results.append((frame.filename, Status.NOT_CHECKED, f"no raw image type for mode {mode.key}"))
            continue
        try:
            with frame.open() as hdulist:
                status, message = check_object(obsres.instrument, hdulist, astype=mode.rawimage)
        except OSError as error:
            status, message = Status.INVALID, error_message(error)
        results.append((frame.filename, status, message))

    if any(status is Status.INVALID for _, status, _ in results):
        results.append((obname, Status.NOT_CHECKED, "some raw images are not valid"))
    elif mode.validator is None:
        results.append((obname, Status.NOT_CHECKED, f"no validator for mode {mode.key}"))
    else:
        try:
            mode.validator(mode, obsres)
            results.append((obname, Status.VALID, ""))
        except Exception as error:
            results.append((obname, Status.INVALID, error_message(error)))
    return results
