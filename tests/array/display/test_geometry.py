"""Window geometry is a tuple (x, y, dx, dy) in the whole package"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numina.array.display.matplotlib_qt import parse_geometry, set_window_geometry
from numina.array.display.ximshow import ximshow


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


def test_ximshow_geometry_tuple():
    # with the Agg backend only the figure size is set (dx, dy in pixels)
    ax = ximshow(np.arange(100.0).reshape(10, 10), geometry=(0, 0, 1024, 768), show=False)
    try:
        assert tuple(ax.figure.get_size_inches()) == pytest.approx((10.24, 7.68))
    finally:
        plt.close(ax.figure)


def test_ximshow_geometry_string_is_rejected():
    with pytest.raises(TypeError, match="not a string"):
        ximshow(np.arange(100.0).reshape(10, 10), geometry="1024,768,0,0", show=False)
    plt.close("all")


def test_ximshow_factor_macosx_is_deprecated():
    with pytest.warns(DeprecationWarning, match="factor_macosx"):
        ax = ximshow(np.arange(100.0).reshape(10, 10), factor_macosx=1.5, show=False)
    plt.close(ax.figure)
