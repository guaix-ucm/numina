#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Download of files, with a cache and a check of their hash

The files are downloaded with :func:`astropy.utils.data.download_file`,
in the cache of astropy for numina (``~/.cache/numina/download`` in linux).
The functions of :mod:`astropy.utils.data` manage the cache, with
``pkgname="numina"``: for example, ``export_download_cache`` and
``import_download_cache`` copy it to computers without internet access.
"""

import hashlib

#: The name of the cache of astropy used by numina
CACHE_NAME = "numina"

#: The algorithms of the hashes
HASH_ALGORITHMS = ("md5", "sha1", "sha256")


def file_hash(path, algorithm="sha256"):
    """The hash of a file, as a hexadecimal string"""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path, known_hash):
    """Check the hash of a file, `known_hash` is "algorithm:value"

    Raises
    ------
    ValueError
        If the hash is not valid, or it is not the hash of the file.
    """
    algorithm, sep, value = known_hash.partition(":")
    if not sep or algorithm not in HASH_ALGORITHMS:
        raise ValueError(f"invalid hash {known_hash}, use algorithm:value, with algorithm in {HASH_ALGORITHMS}")
    computed = file_hash(path, algorithm)
    if computed != value.lower():
        raise ValueError(f"the {algorithm} hash of the file is {computed}, not {value}")


def download_file(url, known_hash=None, cache=True, show_progress=True):
    """Download a file, return the path of the downloaded file

    The file is kept in the cache of numina, and it is not downloaded
    again. If `known_hash` is given ("algorithm:value"), the hash of the
    file is checked, and a file with other hash is removed from the cache.

    Raises
    ------
    ValueError
        If the hash of the file is not `known_hash`.
    """
    from astropy.utils import data

    path = data.download_file(url, cache=cache, pkgname=CACHE_NAME, show_progress=show_progress)
    if known_hash is not None:
        try:
            check_hash(path, known_hash)
        except ValueError as error:
            if cache:
                data.clear_download_cache(url, pkgname=CACHE_NAME)
            raise ValueError(f"{url}: {error}") from error
    return path
