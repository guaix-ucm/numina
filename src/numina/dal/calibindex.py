#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Index of the installed calibration files, from their envelopes

The files with envelope (see :mod:`numina.store.envelope`) under a
directory are recorded in a registry kept in memory, so that they are
selected as the products of a registry of reductions: by instrument,
type, profile and tags, whatever the directory where the file is. The
files without envelope are not indexed, they are found by their path.
"""

import logging
import os

from numina.exceptions import ValidationError
from numina.store.envelope import read_envelope

from .registry import MemoryStore, Registry

_logger = logging.getLogger(__name__)


def calibration_files(directory):
    """The files under `directory`, sorted, without hidden files and directories"""
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            if not name.startswith("."):
                yield os.path.join(root, name)


def index_calibrations(directory):
    """A registry in memory with the files with envelope under `directory`

    A file whose envelope cannot be read or is not valid is not
    indexed, with a warning.
    """
    registry = Registry(None, basedir=directory, store=MemoryStore())
    for path in calibration_files(directory):
        try:
            envelope = read_envelope(path)
        except (OSError, UnicodeDecodeError, ValueError, ValidationError) as error:
            _logger.warning("calibration file %s not indexed: %s", path, error)
            continue
        if envelope is None:
            continue
        prod_id = registry.add_installed(envelope, path)
        _logger.debug("indexed %s %s, id=%d", envelope["type"], path, prod_id)
    _logger.info("%d calibration files with envelope in %s", len(registry.products()), directory)
    return registry
