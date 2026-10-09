#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import sys
import os
import pytest

from numina.testing.testcache import user_cache_dir


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="runs only in linux")
def test_user_cache_dir_linux_home(monkeypatch, tmpdir):

    home = tmpdir.mkdir("hometest")

    modenviron = {"HOME": home.strpath}

    monkeypatch.setattr(os, "environ", modenviron)

    cachedir = os.path.join(home.strpath, ".cache")
    expected = os.path.join(cachedir, "numina")

    if not os.path.exists(expected):
        os.makedirs(expected)

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected, "astropy"))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="runs only in linux")
def test_user_cache_dir_linux_xdg(monkeypatch, tmpdir):

    home = tmpdir.mkdir("hometest")
    cache = tmpdir.mkdir(".cache")

    modenviron = {"HOME": home.strpath, "XDG_CACHE_HOME": cache.strpath}

    monkeypatch.setattr(os, "environ", modenviron)

    cachedir = cache.strpath
    expected = os.path.join(cachedir, "numina")

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected, "astropy"))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="runs only in linux")
def test_user_cache_dir_linux_xdg_with_appname_inside(monkeypatch, tmpdir):
    """appname in the middle of XDG_CACHE_HOME does not truncate the path"""

    home = tmpdir.mkdir("hometest")
    cache = tmpdir.mkdir("numina-work").mkdir(".cache")

    modenviron = {"HOME": home.strpath, "XDG_CACHE_HOME": cache.strpath}

    monkeypatch.setattr(os, "environ", modenviron)

    expected = os.path.join(cache.strpath, "numina")

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected, "astropy"))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="runs only in linux")
def test_user_cache_dir_linux_xdg_ends_with_appname(monkeypatch, tmpdir):
    """XDG_CACHE_HOME set by download_cache is not extended again"""

    home = tmpdir.mkdir("hometest")
    cache = tmpdir.mkdir(".cache").mkdir("numina")

    modenviron = {"HOME": home.strpath, "XDG_CACHE_HOME": cache.strpath}

    monkeypatch.setattr(os, "environ", modenviron)

    assert user_cache_dir("numina") == cache.strpath
    assert os.path.exists(os.path.join(cache.strpath, "astropy"))


@pytest.mark.skipif(sys.platform != "darwin", reason="runs only in darwin")
def test_user_cache_dir_darwin_home(monkeypatch, tmpdir):

    home = tmpdir.mkdir("hometest")

    modenviron = {"HOME": home.strpath}

    monkeypatch.setattr(os, "environ", modenviron)

    cachedir = os.path.join(home.strpath, "Library/Caches")
    expected = os.path.join(cachedir, "numina")

    if not os.path.exists(expected):
        os.makedirs(expected)

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected, "astropy"))


@pytest.mark.skipif(sys.platform != "darwin", reason="runs only in darwin")
def test_user_cache_dir_darwin_xdg(monkeypatch, tmpdir):

    home = tmpdir.mkdir("hometest")
    cache = tmpdir.mkdir(".cache")

    modenviron = {"HOME": home.strpath, "XDG_CACHE_HOME": cache.strpath}

    monkeypatch.setattr(os, "environ", modenviron)

    cachedir = os.path.join(home.strpath, "Library/Caches")
    expected = os.path.join(cachedir, "numina")

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected, "astropy"))
