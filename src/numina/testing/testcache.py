#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Global cache for testing."""

import os
import sys
import tempfile

import numina.util.context as cntx

from astropy.utils import data


def user_cache_dir(appname=None):
    """Directory of the cache used to download test data

    In linux, it is $XDG_CACHE_HOME/appname or ~/.cache/appname.
    If XDG_CACHE_HOME already ends with appname, as it does
    inside download_cache, it is used as is.
    """
    if sys.platform == "darwin":
        path = os.path.expanduser("~/Library/Caches")
    else:
        path = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    if appname and os.path.basename(os.path.normpath(path)) != appname:
        path = os.path.join(path, appname)
    os.makedirs(os.path.join(path, "astropy"), exist_ok=True)
    return path


def download_cache(url, cache=True):
    """Get a tempfile from an URL"""
    cache_dir = user_cache_dir("numina")

    with cntx.environ(XDG_CACHE_HOME=cache_dir):
        with open(data.download_file(url, cache=cache), "rb") as fs:
            with tempfile.NamedTemporaryFile(delete=False) as fd:
                block = fs.read()
                while block:
                    fd.write(block)
                    block = fs.read()

        return fd
