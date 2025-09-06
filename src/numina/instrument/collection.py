#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import itertools
import json
import os
import pathlib
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


import attrs

import importlib_resources

from .configorigin import ElementOrigin


FileLike = str | os.PathLike


@attrs.define
class ComponentCollection:
    dirname = attrs.field()
    paths = attrs.field()


def load_paths_store(
    pkg_paths: "Iterable[FileLike] | None" = None,
    file_paths: "Iterable[FileLike] | None" = None,
) -> dict[str, typing.Any]:
    comp_store = {}
    # Prepare file paths
    if file_paths is None:
        file_paths = []
    if pkg_paths is None:
        pkg_paths = []

    paths1 = [pathlib.Path(f_path) for f_path in file_paths]
    paths2 = [importlib_resources.files(p_path) for p_path in pkg_paths]

    for path in itertools.chain(paths1, paths2):
        for obj in path.iterdir():
            if obj.suffix == ".json":
                with open(obj) as fd:
                    cont = json.load(fd)
                    cont["origin"] = ElementOrigin.from_dict(cont)
                    comp_store[obj.name] = cont

    return comp_store
