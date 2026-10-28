"""Fixtures of numina.testing.plugins that download datasets

The datasets are tarballs created by the tests, download_cache is replaced
by a copy of them.
"""

import io
import os
import tarfile
import tempfile

import pytest
import yaml

import numina.testing.plugins as plugins

from .user import testdrp

# numinatpldir
BASE_URL = "https://example.com/datasets/"
# datamanager_remote
TEST_SET_HOST = BASE_URL
TEST_SET_FILE = "session.tar.gz"
TEST_SET_OBS = "bias.yaml"


def add_file(tar, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    base = tmp_path_factory.mktemp("datasets")
    # The dataset of numinatpldir is named after the test function
    with tarfile.open(base / "numinatpldir.tar.gz", "w:gz") as tar:
        add_file(tar, "tpl/data.txt", b"data")
    with tarfile.open(base / "session.tar.gz", "w:gz") as tar:
        add_file(tar, "data/README", b"")
        add_file(tar, "control.yaml", yaml.safe_dump({"version": 1}).encode())
        add_file(tar, "bias.yaml", yaml.safe_dump(testdrp.BIAS).encode())
    return base


@pytest.fixture(scope="module", autouse=True)
def downloads(datasets):
    """The URLs downloaded, download_cache copies the datasets"""
    urls = []

    def download_cache(url, cache=True):
        urls.append(url)
        with tempfile.NamedTemporaryFile(delete=False) as fd:
            fd.write((datasets / url.removeprefix(BASE_URL)).read_bytes())
        return fd

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plugins, "download_cache", download_cache)
        yield urls


@pytest.fixture
def extracted(monkeypatch):
    """The keyword arguments of the calls to TarFile.extractall"""
    calls = []
    extractall = tarfile.TarFile.extractall

    def record(self, *args, **kwargs):
        calls.append(kwargs)
        return extractall(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extractall", record)
    return calls


@pytest.fixture
def keep_cwd(monkeypatch):
    # numinatpldir changes the working directory
    monkeypatch.chdir(os.getcwd())


def test_numinatpldir(keep_cwd, numinatpldir, downloads):
    assert downloads[-1] == BASE_URL + "numinatpldir.tar.gz"
    assert os.getcwd() == str(numinatpldir / "tpl")
    assert (numinatpldir / "tpl" / "data.txt").read_text("utf-8") == "data"


@pytest.mark.xfail(reason="extractall without filter warns in Python 3.12 and 3.13", strict=True)
def test_numinatpldir_filter(keep_cwd, extracted, numinatpldir):
    assert extracted == [{"filter": "data"}]


def test_datamanager_remote(datamanager_remote, downloads):
    assert BASE_URL + TEST_SET_FILE in downloads
    assert set(datamanager_remote.backend.ob_table) == {testdrp.BIAS["id"]}


def test_numinatmpdir(tmpdir):
    # The fixture is deprecated, it is called directly to check the warning
    with pytest.warns(DeprecationWarning, match="numinatmpdir"):
        result = plugins.numinatmpdir.__wrapped__(tmpdir)
    assert (result / "_work").isdir()
    assert (result / "_data").isdir()


@pytest.mark.parametrize("has_benchmark, header", [(True, ""), (False, "pytest-benchmark not installed")])
def test_report_header(monkeypatch, has_benchmark, header):
    monkeypatch.setattr(plugins, "HAS_BENCHMARCK", has_benchmark)
    assert plugins.pytest_report_header(None) == header
