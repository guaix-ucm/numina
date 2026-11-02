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

The envelopes are kept in a cache, a JSON file for each directory and
version of numina in the cache of numina (``~/.cache/numina/calibindex``
in linux, see :func:`default_cache_dir`), so that environments with
other versions of numina do not discard the cache of each other. Only
the files that are new, or whose size or modification time has
changed, are read again. When a cache is written, the caches of
directories that no longer exist, and those not used in
:data:`CACHE_MAX_AGE` days, are removed.
"""

import hashlib
import json
import logging
import os
import time

import numina
from numina.exceptions import ValidationError
from numina.store.envelope import read_envelope

from .registry import MemoryStore, Registry, atomic_write

_logger = logging.getLogger(__name__)

#: Value of the key ``format`` of the cache files
CACHE_FORMAT = "numina-calibindex"
#: Version of the cache files
CACHE_VERSION = 1
#: Days after which a cache that has not been used is removed
CACHE_MAX_AGE = 30


def default_cache_dir():
    """The directory of the caches of the indexes, ``calibindex`` in the cache of numina"""
    from astropy.config.paths import get_cache_dir_path

    return os.path.join(get_cache_dir_path(rootname="numina"), "calibindex")


def cache_path(directory, cache_dir):
    """The cache file of the index of `directory`, named by the hash of its absolute path and the version of numina"""
    key = f"{os.path.abspath(directory)}\0{numina.__version__}"
    name = hashlib.sha256(key.encode()).hexdigest()[:32]
    return os.path.join(cache_dir, f"{name}.json")


def calibration_files(directory):
    """The files under `directory`, sorted, without hidden files and directories"""
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            if not name.startswith("."):
                yield os.path.join(root, name)


def _load_cache(path, directory):
    """The entries of the cache file `path` of `directory`, by relative path; empty if it is not valid"""
    try:
        with open(path) as fd:
            data = json.load(fd)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        _logger.debug("cache %s not used: %s", path, error)
        return {}
    expected = (CACHE_FORMAT, CACHE_VERSION, numina.__version__, os.path.abspath(directory))
    found = tuple(data.get(key) for key in ("format", "version", "numina", "directory"))
    if found != expected:
        _logger.debug("cache %s not used, it is for %s, not %s", path, found, expected)
        return {}
    return data.get("files", {})


def _write_cache(path, directory, entries):
    """Write the cache file `path` of `directory`, and remove the caches of directories that do not exist"""
    data = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "numina": numina.__version__,
        "directory": os.path.abspath(directory),
        "files": entries,
    }
    cache_dir = os.path.dirname(path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        atomic_write(path, lambda fp: json.dump(data, fp))
    except OSError as error:
        _logger.debug("cache %s not written: %s", path, error)
        return
    _logger.debug("cache of %s written in %s", directory, path)
    _prune_caches(cache_dir)


def _prune_caches(cache_dir):
    """Remove the cache files of `cache_dir` whose directory does not exist, or not used in CACHE_MAX_AGE days"""
    oldest = time.time() - CACHE_MAX_AGE * 86400
    for name in os.listdir(cache_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(cache_dir, name)
        try:
            if os.path.getmtime(path) < oldest:
                os.remove(path)
                _logger.debug("cache %s removed, not used in %d days", path, CACHE_MAX_AGE)
                continue
            with open(path) as fd:
                directory = json.load(fd).get("directory")
            if directory and not os.path.isdir(directory):
                os.remove(path)
                _logger.debug("cache %s of %s removed, the directory does not exist", path, directory)
        except (OSError, ValueError, AttributeError) as error:
            _logger.debug("cache %s not checked: %s", path, error)


def _touch(path):
    """Update the modification time of the cache `path`, the time it was last used"""
    try:
        os.utime(path)
    except OSError as error:
        _logger.debug("cache %s not touched: %s", path, error)


def _read_entry(path, size, mtime_ns):
    """The entry of the cache for the file `path`: its size, time, envelope and error"""
    entry = {"size": size, "mtime_ns": mtime_ns, "envelope": None, "error": None}
    try:
        entry["envelope"] = read_envelope(path)
    except (OSError, UnicodeDecodeError, ValueError, ValidationError) as error:
        entry["error"] = str(error)
    return entry


def read_envelopes(directory, cache_dir=None, cache=True):
    """The envelopes of the files under `directory`, a dict by path of (envelope, error)

    The envelope is None for the files without envelope, and the error
    is the message of the error raised when the envelope was read, or
    None. If `cache` is True, the envelopes are kept in a cache in
    `cache_dir` (by default, :func:`default_cache_dir`), and only the
    files that are new or have changed are read.
    """
    if cache:
        cache_dir = default_cache_dir() if cache_dir is None else cache_dir
        path_cache = cache_path(directory, cache_dir)
        cached = _load_cache(path_cache, directory)
    else:
        cached = {}
    entries = {}
    nread = 0
    for path in calibration_files(directory):
        relpath = os.path.relpath(path, directory)
        try:
            stat = os.stat(path)
        except OSError as error:
            entries[relpath] = {"size": None, "mtime_ns": None, "envelope": None, "error": str(error)}
            continue
        entry = cached.get(relpath)
        if entry is None or entry.get("size") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
            entry = _read_entry(path, stat.st_size, stat.st_mtime_ns)
            nread += 1
        entries[relpath] = entry
    _logger.debug("%d of %d files of %s read, the others from the cache", nread, len(entries), directory)
    if cache and (nread or set(entries) != set(cached)):
        _write_cache(path_cache, directory, entries)
    elif cache:
        _touch(path_cache)
    return {os.path.join(directory, rel): (entry["envelope"], entry["error"]) for rel, entry in entries.items()}


def index_calibrations(directory, cache_dir=None, cache=True):
    """A registry in memory with the files with envelope under `directory`

    A file whose envelope cannot be read or is not valid is not
    indexed, with a warning. The envelopes are read with
    :func:`read_envelopes`, with the cache in `cache_dir` if `cache`
    is True.
    """
    registry = Registry(None, basedir=directory, store=MemoryStore())
    envelopes = read_envelopes(directory, cache_dir=cache_dir, cache=cache)
    # One transaction, each transaction copies the store
    with registry.store.transaction():
        for path, (envelope, error) in envelopes.items():
            if error is not None:
                _logger.warning("calibration file %s not indexed: %s", path, error)
                continue
            if envelope is None:
                continue
            prod_id = registry.add_installed(envelope, path)
            _logger.debug("indexed %s %s, id=%d", envelope["type"], path, prod_id)
    _logger.info("%d calibration files with envelope in %s", len(registry.products()), directory)
    return registry
