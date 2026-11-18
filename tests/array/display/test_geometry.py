"""Window geometry is a tuple (x, y, dx, dy) in the whole package"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytest

import numina.array.display.ximshow as ximshow_module
from numina.array.display.matplotlib_utils import parse_geometry, set_window_geometry
from numina.array.display.ximshow import ximshow


@pytest.fixture
def geometries(monkeypatch):
    """Geometries that ximshow passes to set_window_geometry, whatever the backend"""
    calls = []
    monkeypatch.setattr(ximshow_module, "set_window_geometry", calls.append)
    return calls


def test_parse_geometry():
    assert parse_geometry("0,0,1024,768") == (0, 0, 1024, 768)
    assert parse_geometry(" 10, 20 ,640,480") == (10, 20, 640, 480)


@pytest.mark.parametrize("text", ["0,0,640", "0,0,640,480,1", "0,0,640.5,480", "a,b,c,d", ""])
def test_parse_geometry_invalid(text):
    with pytest.raises(argparse.ArgumentTypeError, match="expected 4 integers"):
        parse_geometry(text)


def test_parse_geometry_in_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="0,0,640,480", type=parse_geometry)
    assert parser.parse_args([]).geometry == (0, 0, 640, 480)
    assert parser.parse_args(["--geometry", "1,2,3,4"]).geometry == (1, 2, 3, 4)


def test_set_window_geometry_rejects_strings():
    with pytest.raises(TypeError, match="not a string"):
        set_window_geometry("1024,768,0,0")


def test_ximshow_geometry_tuple(geometries):
    ax = ximshow(np.arange(100.0).reshape(10, 10), geometry=(0, 0, 1024, 768), show=False)
    plt.close(ax.figure)
    assert geometries == [(0, 0, 1024, 768)]


def test_ximshow_geometry_string_is_deprecated(geometries):
    """A string is read with the convention before numina 0.42, 'xwidth,ywidth,xorigin,yorigin'"""
    with pytest.warns(DeprecationWarning, match="string geometry"):
        ax = ximshow(np.arange(100.0).reshape(10, 10), geometry="1024,768,10,20", show=False)
    plt.close(ax.figure)
    assert geometries == [(10, 20, 1024, 768)]


def test_ximshow_geometry_string_without_size_is_ignored(geometries):
    with pytest.warns(DeprecationWarning, match="string geometry"):
        ax = ximshow(np.arange(100.0).reshape(10, 10), geometry="0,0,640,480", show=False)
    plt.close(ax.figure)
    assert geometries == [None]


def test_ximshow_geometry_string_invalid():
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="xwidth,ywidth,xorigin,yorigin"):
        ximshow(np.arange(100.0).reshape(10, 10), geometry="1024,768", show=False)
    plt.close("all")


def test_ximshow_factor_macosx_is_deprecated():
    with pytest.warns(DeprecationWarning, match="factor_macosx"):
        ax = ximshow(np.arange(100.0).reshape(10, 10), factor_macosx=1.5, show=False)
    plt.close(ax.figure)
