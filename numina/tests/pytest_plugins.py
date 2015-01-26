#
# Copyright 2014-2015 Universidad Complutense de Madrid
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
import tarfile

import pytest

from .testcache import download_cache


@pytest.fixture
def numinatmpdir(tmpdir):
    '''return a temporary directory path object
    for numina, derived from tmpdir
    '''
    tmpdir.mkdir('_work')
    tmpdir.mkdir('_data')
    return tmpdir


@pytest.fixture
def numinatpldir(tmpdir, request):
    '''return a temporary directory path object
    for numina, where a dataset has been downloaded
    from a remote location, based on
    the module variable BASE_URL and the test function name
    '''

    # Name of the dataset based on the function name
    tarname = request.function.__name__[5:]
    # Base url to donwload
    base = getattr(request.module, 'BASE_URL')
    url = base + tarname + '.tar.gz'

    downloaded = download_cache(url)

    tmpdir.chdir()

    # Uncompress
    with tarfile.open(downloaded.name, mode="r:gz") as tar:
        tar.extractall()

    os.remove(downloaded.name)

    os.chdir('tpl')

    return tmpdir


def pytest_addoption(parser):
    parser.addoption("--run-remote", action="store_true", default=False,
                     help="run tests with online data")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers",
        "remote: mark test to run with online data"
        )


def pytest_runtest_setup(item):
    if ('remote' in item.keywords and
            not item.config.getoption("--run-remote")):

        pytest.skip("need --run-remote option to run")
