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

from .configorigin import ElementOrigin

FileLike = str | os.PathLike

_logger = logging.getLogger(__name__)


@attrs.define
class ComponentCollection:
    dirname = attrs.field()
    paths = attrs.field()


def load_paths_store(
    pkg_paths: "Iterable[str] | None" = None,
    file_paths: "Iterable[FileLike] | None" = None,
) -> dict[str, typing.Any]:
    comp_store = {}
    # Prepare file paths
    if file_paths is None:
        file_paths = []
    if pkg_paths is None:
        pkg_paths = []

    paths1 = [pathlib.Path(f_path) for f_path in file_paths]
    paths2 = [importlib.resources.files(p_path) for p_path in pkg_paths]

    # Directory where each file was read
    file_dirs = {}
    for path in itertools.chain(paths1, paths2):
        for obj in path.iterdir():
            if obj.suffix == ".json":
                with obj.open() as fd:
                    cont = json.load(fd)
                    cont["origin"] = ElementOrigin.from_dict(cont)
                    if obj.name in comp_store:
                        _logger.warning(
                            "configuration file %s in %s replaces the one in %s", obj.name, path, file_dirs[obj.name]
                        )
                    comp_store[obj.name] = cont
                    file_dirs[obj.name] = path

    return comp_store
