import os

import astropy.io.fits as fits
import numpy
import pytest

from numina.types.dataframe import DataFrame
from numina.util.context import environ, ignored, manage_fits, working_directory


def test_environ_context1():
    """Remove env var if it was not set"""

    cache_dir = "/cache/dir"

    with environ(OOO_CACHE_HOME=cache_dir):
        assert os.environ["OOO_CACHE_HOME"] == cache_dir

    is_in = "OOO_CACHE_HOME" in os.environ

    assert is_in is False


def test_environ_context2():
    """Reset env var if it was set"""

    cache_dir1 = "/cache/dir/1"
    cache_dir2 = "/cache/dir/2"

    os.environ["OOO_CACHE_HOME"] = cache_dir1

    with environ(OOO_CACHE_HOME=cache_dir2):
        assert os.environ["OOO_CACHE_HOME"] == cache_dir2

    assert os.environ["OOO_CACHE_HOME"] == cache_dir1


def test_environ_context3():
    """Reset multiple variables"""

    cache_dir1a = "/cache/dir/1a"
    cache_dir2a = "/cache/dir/2a"
    cache_dir3a = "/cache/dir/3a"
    cache_dir1b = "/cache/dir/1b"
    cache_dir2b = "/cache/dir/2b"

    os.environ["OOO_CACHE_HOME1"] = cache_dir1a
    os.environ["OOO_CACHE_HOME2"] = cache_dir2a
    os.environ["OOO_CACHE_HOME3"] = cache_dir3a

    with environ(OOO_CACHE_HOME1=cache_dir1b, OOO_CACHE_HOME2=cache_dir2b):
        assert os.environ["OOO_CACHE_HOME1"] == cache_dir1b
        assert os.environ["OOO_CACHE_HOME2"] == cache_dir2b
        assert os.environ["OOO_CACHE_HOME3"] == cache_dir3a

    assert os.environ["OOO_CACHE_HOME1"] == cache_dir1a
    assert os.environ["OOO_CACHE_HOME2"] == cache_dir2a
    assert os.environ["OOO_CACHE_HOME3"] == cache_dir3a


def test_working_directory(tmp_path):
    cwd = os.getcwd()
    with working_directory(tmp_path):
        assert os.getcwd() == str(tmp_path)
    assert os.getcwd() == cwd

    with pytest.raises(RuntimeError):
        with working_directory(tmp_path):
            raise RuntimeError
    assert os.getcwd() == cwd


def test_ignored():
    with ignored(KeyError, IndexError):
        {}["a"]

    with pytest.raises(ValueError):
        with ignored(KeyError):
            raise ValueError


class Resource:
    closed = False

    def close(self):
        self.closed = True


def test_manage_fits(tmp_path):
    filename = tmp_path / "image.fits"
    fits.PrimaryHDU(numpy.ones((2, 2))).writeto(filename)
    hdul = fits.HDUList([fits.PrimaryHDU(numpy.zeros((2, 2)))])
    frame = DataFrame(filename=str(filename))
    resource = Resource()

    with manage_fits([str(filename), hdul, frame, resource]) as refs:
        assert len(refs) == 4
        assert numpy.all(refs[0][0].data == 1)
        assert refs[1] is hdul
        assert numpy.all(refs[2][0].data == 1)
        assert refs[3] is resource

    assert resource.closed
    assert refs[0].fileinfo(0)["file"].closed
    assert refs[2].fileinfo(0)["file"].closed
