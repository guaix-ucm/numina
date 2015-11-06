#
# Copyright 2015 Universidad Complutense de Madrid
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

import os

import pytest

from .testcache import user_cache_dir


@pytest.mark.xfail(reason='issue #98')
def test_user_cache_dir_linux_home(monkeypatch, tmpdir):

    home = tmpdir.mkdir('hometest')

    modenviron = {'HOME': home.strpath}

    monkeypatch.setattr(os, 'environ', modenviron)

    cachedir = os.path.join(home.strpath, '.cache')
    expected = os.path.join(cachedir, 'numina', 'astropy')

    assert user_cache_dir("numina") == expected


@pytest.mark.xfail(reason="issue #99")
def test_user_cache_dir_linux_xdg(monkeypatch, tmpdir):

    home = tmpdir.mkdir('hometest')
    cache = tmpdir.mkdir('cache')

    modenviron = {'HOME': home.strpath,
                  'XDG_CACHE_HOME': cache.strpath
                  }

    monkeypatch.setattr(os, 'environ', modenviron)

    cachedir = cache.strpath
    expected = os.path.join(cachedir, 'numina', 'astropy')

    assert user_cache_dir("numina") == expected
