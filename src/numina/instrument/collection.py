#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import importlib.resources
import itertools
import json
import logging
import os
import pathlib
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


import attrs

from numina.schemas import validate as validate_schema
from .configorigin import ElementOrigin

FileLike = str | os.PathLike

_logger = logging.getLogger(__name__)


@attrs.define
class ComponentCollection:
    """Record with the attributes ``dirname`` and ``paths``."""

    dirname = attrs.field()
    paths = attrs.field()


def load_paths_store(
    pkg_paths: "Iterable[str] | None" = None,
    file_paths: "Iterable[FileLike] | None" = None,
) -> dict[str, typing.Any]:
    """Load the JSON configuration files of several directories.

    Each ``.json`` file is validated with the schema ``instconf``, and its ``origin`` key
    is set to the ``ElementOrigin`` built from its contents. The directories in ``file_paths``
    are read before the packages in ``pkg_paths``, and have priority: a file with the name of
    a file read before is ignored, with a warning. So the configurations of ``file_paths`` replace
    those of the packages with the same file name, and are the first in the collection (the first
    is used when several are valid for a date, see ``find_element``).

    Parameters
    ----------
    pkg_paths
        Names of the packages with configuration files.
    file_paths
        Directories with configuration files.

    Returns
    -------
    dict
        The contents of the files, by file name.
    """
    comp_store = {}
    # Prepare file paths
    if file_paths is None:
        file_paths = []
    if pkg_paths is None:
        pkg_paths = []

    # The name of each directory or package, and its path
    paths1 = [(str(f_path), pathlib.Path(f_path)) for f_path in file_paths]
    paths2 = [(p_path, importlib.resources.files(p_path)) for p_path in pkg_paths]

    # Directory where each file was read
    file_dirs = {}
    for label, path in itertools.chain(paths1, paths2):
        for obj in path.iterdir():
            if obj.suffix == ".json":
                if obj.name in comp_store:
                    # The first file read has priority
                    _logger.warning(
                        "configuration file %s in %s is ignored, the one in %s is used",
                        obj.name,
                        label,
                        file_dirs[obj.name],
                    )
                    continue
                with obj.open() as fd:
                    cont = json.load(fd)
                    validate_schema(cont, "instconf", source=str(obj))
                    cont["origin"] = ElementOrigin.from_dict(cont)
                    comp_store[obj.name] = cont
                    file_dirs[obj.name] = label

    return comp_store
