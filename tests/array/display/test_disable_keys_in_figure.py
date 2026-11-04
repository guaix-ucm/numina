"""Tests of disable_keys_in_figure and restore_default_keys"""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure

from numina.array.display.disable_keys_in_figure import disable_keys_in_figure, restore_default_keys


@pytest.fixture
def fig():
    """Pyplot figure with one axes, closed after the test"""
    figure, ax = plt.subplots()
    ax.plot([1, 2, 3])
    yield figure
    plt.close(figure)


def press(figure, key):
    """Simulate pressing a key with the mouse over the centre of the axes"""
    ax = figure.axes[0]
    x, y = ax.transAxes.transform((0.5, 0.5))
    figure.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", figure.canvas, key, x, y))


def n_key_handlers(figure):
    """Number of callbacks connected to key_press_event"""
    return len(figure.canvas.callbacks.callbacks.get("key_press_event", {}))


# The default keymaps toggle the y scale with 'l' and the x scale with 'k'


def test_default_keys_work(fig):
    press(fig, "l")
    assert fig.axes[0].get_yscale() == "log"


def test_disabled_key_is_ignored_and_other_keys_work(fig):
    disable_keys_in_figure(fig, ["l"])
    press(fig, "l")
    assert fig.axes[0].get_yscale() == "linear"
    press(fig, "k")
    assert fig.axes[0].get_xscale() == "log"


def test_single_string_is_one_key(fig):
    # Previously a string was split into its characters
    disable_keys_in_figure(fig, "ctrl+l")
    press(fig, "l")
    assert fig.axes[0].get_yscale() == "log"


def test_keys_are_case_sensitive(fig):
    # 'L' toggles the x scale in the default keymap, and is not blocked by 'l'
    assert "L" in mpl.rcParams["keymap.xscale"]
    disable_keys_in_figure(fig, ["l"])
    press(fig, "L")
    assert fig.axes[0].get_xscale() == "log"


def test_successive_calls_accumulate(fig):
    disable_keys_in_figure(fig, ["l"])
    disable_keys_in_figure(fig, ["k"])
    press(fig, "l")
    press(fig, "k")
    assert fig.axes[0].get_yscale() == "linear"
    assert fig.axes[0].get_xscale() == "linear"


def test_restore_default_keys(fig):
    disable_keys_in_figure(fig, ["l", "k"])
    restore_default_keys(fig)
    press(fig, "l")
    assert fig.axes[0].get_yscale() == "log"
    # after restoring, disabling starts again from an empty set of keys
    disable_keys_in_figure(fig, ["k"])
    press(fig, "l")
    assert fig.axes[0].get_yscale() == "linear"


def test_restore_without_disabling(fig):
    restore_default_keys(fig)
    press(fig, "l")
    assert fig.axes[0].get_yscale() == "log"


def test_handlers_are_replaced_not_added(fig):
    n = n_key_handlers(fig)
    cid = disable_keys_in_figure(fig, ["l"])
    assert fig.canvas.manager.key_press_handler_id == cid
    disable_keys_in_figure(fig, ["k"])
    assert n_key_handlers(fig) == n
    cid = restore_default_keys(fig)
    assert fig.canvas.manager.key_press_handler_id == cid
    assert n_key_handlers(fig) == n


def test_other_figures_are_not_affected(fig):
    other, ax = plt.subplots()
    try:
        disable_keys_in_figure(fig, ["l"])
        press(other, "l")
        assert ax.get_yscale() == "log"
    finally:
        plt.close(other)


@pytest.mark.parametrize("function", [lambda f: disable_keys_in_figure(f, ["l"]), restore_default_keys])
def test_figure_without_manager_raises(function):
    figure = Figure()
    figure.add_subplot()
    with pytest.raises(ValueError, match="canvas manager"):
        function(figure)


@pytest.mark.parametrize("function", [lambda f: disable_keys_in_figure(f, ["l"]), restore_default_keys])
def test_toolmanager_raises(function):
    # Previously the standard handler was added on top of the ToolManager
    with warnings.catch_warnings():
        # matplotlib warns that the ToolManager is experimental
        warnings.simplefilter("ignore", UserWarning)
        with mpl.rc_context({"toolbar": "toolmanager"}):
            figure = plt.figure()
    try:
        n = n_key_handlers(figure)
        with pytest.raises(NotImplementedError, match="toolmanager"):
            function(figure)
        assert n_key_handlers(figure) == n
    finally:
        plt.close(figure)
