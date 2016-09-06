#
# Copyright 2015-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import os
import pytest

from .testcache import user_cache_dir

@pytest.mark.skipif(sys.platform != 'linux2',
                    reason="runs only in linux")
def test_user_cache_dir_linux_home(monkeypatch, tmpdir):

    home = tmpdir.mkdir('hometest')

    modenviron = {'HOME': home.strpath}

    monkeypatch.setattr(os, 'environ', modenviron)

    cachedir = os.path.join(home.strpath, '.cache')
    expected = os.path.join(cachedir, 'numina')

    if not os.path.exists(expected):
        os.makedirs(expected)

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected,'astropy'))


@pytest.mark.skipif(sys.platform != 'linux2',
                    reason="runs only in linux")
def test_user_cache_dir_linux_xdg(monkeypatch, tmpdir):

    home = tmpdir.mkdir('hometest')
    cache = tmpdir.mkdir('.cache')

    modenviron = {'HOME': home.strpath,
                  'XDG_CACHE_HOME': cache.strpath
                  }

    monkeypatch.setattr(os, 'environ', modenviron)

    cachedir = cache.strpath
    expected = os.path.join(cachedir, 'numina')

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected,'astropy'))


@pytest.mark.skipif(sys.platform != 'darwin',
                    reason="runs only in darwin")
def test_user_cache_dir_darwin_home(monkeypatch, tmpdir):

    home = tmpdir.mkdir('hometest')

    modenviron = {'HOME': home.strpath}

    monkeypatch.setattr(os, 'environ', modenviron)

    cachedir = os.path.join(home.strpath, 'Library/Caches')
    expected = os.path.join(cachedir, 'numina')

    if not os.path.exists(expected):
        os.makedirs(expected)

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected,'astropy'))


@pytest.mark.skipif(sys.platform != 'darwin',
                    reason="runs only in darwin")
def test_user_cache_dir_darwin_xdg(monkeypatch, tmpdir):

    home = tmpdir.mkdir('hometest')
    cache = tmpdir.mkdir('.cache')

    modenviron = {'HOME': home.strpath,
                  'XDG_CACHE_HOME': cache.strpath
                  }

    monkeypatch.setattr(os, 'environ', modenviron)

    cachedir = os.path.join(home.strpath, 'Library/Caches')
    expected = os.path.join(cachedir, 'numina')

    assert user_cache_dir("numina") == expected
    assert os.path.exists(os.path.join(expected,'astropy'))
