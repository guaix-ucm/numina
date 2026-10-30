#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Installation of the datasets of calibrations of the DRPs

A DRP declares its datasets of calibrations in the key ``calibrations``
of its ``drp.yaml``, each one a ZIP file with its URL and its hash. The
files are installed in the directory of calibrations used by
``numina run`` (``calibsdir``), under a directory with the name of the
instrument.
"""

import configparser
import datetime
import logging
import os
import pathlib
import shutil
import tempfile
import zipfile
import zlib

_logger = logging.getLogger(__name__)


class InstallPlan:
    """What the installation of a dataset does with each file

    Each item of the lists is a pair (name of the member of the ZIP file,
    destination path).
    """

    def __init__(self):
        #: Files that do not exist
        self.new = []
        #: Files that exist with the same contents
        self.unchanged = []
        #: Files that exist with other contents
        self.conflicts = []
        #: Members of the ZIP file that are not installed
        self.skipped = []


def find_dataset(drp, name=None):
    """The dataset of calibrations `name` of the DRP, by default the last one

    Raises
    ------
    ValueError
        If the DRP has no datasets, or none with that name.
    """
    datasets = getattr(drp, "calibrations", None) or []
    if not datasets:
        raise ValueError(f"the DRP of {drp.name} has no datasets of calibrations")
    if name is None:
        return datasets[-1]
    for dataset in datasets:
        if dataset["name"] == name:
            return dataset
    names = ", ".join(dataset["name"] for dataset in datasets)
    raise ValueError(f"the DRP of {drp.name} has no dataset {name}, its datasets are {names}")


def _member_parts(name):
    """The parts of the path of a member of a ZIP file, None if it is not safe"""
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.parts


def plan_install(zip_path, destination):
    """The plan to install the files of a ZIP file in `destination`

    If all the files of the ZIP file are under one directory, that
    directory is removed from their paths. The hidden files (a name that
    begins with a dot) and the members with unsafe paths are skipped.
    """
    plan = InstallPlan()
    destination = pathlib.Path(destination)
    with zipfile.ZipFile(zip_path) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        parts = {}
        for info in members:
            member_parts = _member_parts(info.filename)
            if member_parts is None:
                plan.skipped.append((info.filename, None))
                continue
            parts[info.filename] = member_parts
        roots = {member_parts[0] for member_parts in parts.values()}
        strip = len(roots) == 1 and all(len(member_parts) > 1 for member_parts in parts.values())

        for info in members:
            if info.filename not in parts:
                continue
            member_parts = parts[info.filename][1:] if strip else parts[info.filename]
            if member_parts[-1].startswith("."):
                plan.skipped.append((info.filename, None))
                continue
            target = destination.joinpath(*member_parts)
            if not target.exists():
                plan.new.append((info.filename, target))
            elif target.is_file() and target.stat().st_size == info.file_size and _crc32(target) == info.CRC:
                plan.unchanged.append((info.filename, target))
            else:
                plan.conflicts.append((info.filename, target))
    return plan


def _crc32(path):
    """Return the CRC-32 of the contents of the file ``path``."""
    value = 0
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(1 << 20), b""):
            value = zlib.crc32(block, value)
    return value


def extract(zip_path, items):
    """Extract the members of the ZIP file to their destinations

    `items` are pairs (name of the member, destination path). Each file
    is written to a temporary file in its directory and then renamed, so
    that a destination is never left half written.
    """
    with zipfile.ZipFile(zip_path) as zf:
        for name, target in items:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmpname = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
            try:
                with os.fdopen(fd, "wb") as dst, zf.open(name) as src:
                    shutil.copyfileobj(src, dst)
                os.replace(tmpname, target)
            except BaseException:
                os.unlink(tmpname)
                raise


def update_config(cfg_path, base_path, drp, dataset):
    """Record the directory of calibrations and the dataset in a configuration file

    ``calibsdir`` in ``[tool.run]`` is set to `base_path`, and the section
    ``[tool.inittree.<instrument>]`` records the dataset installed.
    """
    import numina

    config = configparser.ConfigParser()
    if os.path.exists(cfg_path):
        config.read(cfg_path)
    if not config.has_section("tool.run"):
        config.add_section("tool.run")
    previous = config["tool.run"].get("calibsdir")
    if previous is not None and previous != str(base_path):
        _logger.warning("calibsdir in %s was %s, now it is %s", cfg_path, previous, base_path)
    config["tool.run"]["calibsdir"] = str(base_path)
    section = f"tool.inittree.{drp.name}"
    config[section] = {
        "dataset": dataset["name"],
        "url": dataset["url"],
        "known_hash": dataset["known_hash"],
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "numina_version": numina.__version__,
    }
    with open(cfg_path, "w") as fd:
        config.write(fd)
