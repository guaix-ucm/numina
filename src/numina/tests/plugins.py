#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import os
import tarfile
import warnings
import sys

import pytest

if "pytest_benchmark" in sys.modules:
    HAS_BENCHMARCK = True
else:
    from .nobenchmark import benchmark
    HAS_BENCHMARCK = False


import numina.util.context as ctx
from .drpmocker import DRPMocker
from .testcache import download_cache
from .pytest_resultcmp import ResultCompPlugin


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


@pytest.fixture(scope='module')
def datamanager_remote(tmp_path_factory, request):
    """Return a DataManager object create from a remote dataset"""
    from numina.user.helpers import create_datamanager

    req_base_default = "https://guaix.fis.ucm.es/data/"
    req_base = getattr(request.module, 'TEST_SET_HOST', req_base_default)
    req_tarname = getattr(request.module, 'TEST_SET_FILE')
    req_datadir = getattr(request.module, 'TEST_SET_DATADIR', 'data')
    req_control = getattr(request.module, 'TEST_SET_CONTROL', "control_v2.yaml")

    basedir = tmp_path_factory.mktemp('manager')

    datadir = basedir / req_datadir  # pathlib syntax
    reqfile  = basedir / req_control

    if req_tarname is None:
        raise ValueError('Undefined TEST_SET_FILE')

    url = req_base + req_tarname

    # Download everything
    with ctx.working_directory(basedir):

        downloaded = download_cache(url)

        # Uncompress
        with tarfile.open(downloaded.name, mode="r:gz") as tar:
            tar.extractall()

        os.remove(downloaded.name)

    # Insert OBS in the control file....
    dm = create_datamanager(reqfile, basedir, datadir)

    # This is not really needed...
    # If everything is in the file already
    # with working_directory(basedir):
    #     obsresults = ['obs_ids.yaml']
    #     sessions, loaded_obs = load_observations(obsresults, is_session=False)
    #    dm.backend.add_obs(loaded_obs)

    return dm


def pytest_report_header(config):
    if not HAS_BENCHMARCK:
        return "pytest-benchmark not installed"
    return ""


def pytest_addoption(parser):
    parser.addoption('--resultcmp', action='store_true',
                    help="enable comparison of recipe results to reference results stored")
    parser.addoption('--resultcmp-generate-path',
                    help="directory to generate reference files in, relative to location where py.test is run", action='store')
    parser.addoption('--resultcmp-reference-path',
                    help="directory containing reference files, relative to location where py.test is run", action='store')


def pytest_configure(config):

    config.getini('markers').append(
        'result_compare: Apply to tests that provide recipe results to compare with a reference')

    if config.getoption("--resultcmp", default=False) or config.getoption("--resultcmp-generate-path", default=None) is not None:

        reference_dir = config.getoption("--resultcmp-reference-path")
        generate_dir = config.getoption("--resultcmp-generate-path")

        if reference_dir is not None and generate_dir is not None:
            warnings.warn("Ignoring --resultcmp-reference-path since --resultcmp-generate-path is set")

        if reference_dir is not None:
            reference_dir = os.path.abspath(reference_dir)
        if generate_dir is not None:
            reference_dir = os.path.abspath(generate_dir)

        # default_format = config.getoption("--resultcmp-default-format") or 'text'
        config.pluginmanager.register(ResultCompPlugin(
            config, reference_dir=reference_dir, generate_dir=generate_dir
        ))
