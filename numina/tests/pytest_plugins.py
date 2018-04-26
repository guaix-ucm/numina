#
# Copyright 2014-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from __future__ import print_function

import os
import tarfile

import pytest
import sys

if "pytest_benchmark" in sys.modules:
    HAS_BENCHMARCK = True
else:
    from .nobenchmark import benchmark
    HAS_BENCHMARCK = False

from .drpmocker import DRPMocker
from .testcache import download_cache


@pytest.fixture
def numinatmpdir(tmpdir):
    """Return a temporary directory for recipe testing"""

    tmpdir.mkdir('_work')
    tmpdir.mkdir('_data')
    return tmpdir


@pytest.fixture
def numinatpldir(tmpdir, request):
    """Return a temporary dataset for recipe testing.

    Return a temporary directory path object
    for numina, where a dataset has been downloaded
    from a remote location, based on
    the module variable BASE_URL and the test function name
    """

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


@pytest.fixture
def drpmocker(monkeypatch):
    """A fixture that mocks the loading of DRPs"""
    return DRPMocker(monkeypatch)


def pytest_addoption(parser):
    parser.addoption("--run-remote", action="store_true", default=False,
                     help="run tests with online data")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers",
        "remote: mark test to run with online data"
        )


def pytest_report_header(config):
    if not HAS_BENCHMARCK:
        return "pytest-benchmark not installed"
    return ""


def pytest_runtest_setup(item):
    if ('remote' in item.keywords and
            not item.config.getoption("--run-remote")):
        
        pytest.skip("need --run-remote option to run")
