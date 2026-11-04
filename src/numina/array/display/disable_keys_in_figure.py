#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Disable and restore the default key bindings of individual matplotlib figures.

Only the default key handler of matplotlib (``key_press_handler``, used
when ``rcParams["toolbar"]`` is not ``"toolmanager"``) is supported.
"""

from matplotlib.backend_bases import key_press_handler

# Name of the figure attribute that stores the keys currently disabled
_DISABLED_KEYS_ATTR = "_numina_disabled_keys"


def _get_manager(fig):
    """Return the canvas manager of a figure with the default key handler.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure created through pyplot.

    Returns
    -------
    manager : matplotlib.backend_bases.FigureManagerBase
        Canvas manager of the figure.

    Raises
    ------
    ValueError
        If the figure has no canvas manager (it was not created through
        pyplot).
    NotImplementedError
        If the figure does not use matplotlib's default key handler
        (``rcParams["toolbar"] == "toolmanager"``, where the keys are
        handled by the ToolManager).
    """
    manager = fig.canvas.manager
    if manager is None:
        raise ValueError("The figure has no canvas manager; it must be created through pyplot")
    if getattr(manager, "toolmanager", None) is not None or manager.key_press_handler_id is None:
        raise NotImplementedError(
            "The figure does not use matplotlib's default key handler "
            '(rcParams["toolbar"] == "toolmanager" is not supported)'
        )
    return manager


def disable_keys_in_figure(fig, keys):
    """Disable selected default key bindings for a single figure.

    The default key handler of `fig` is disconnected and replaced by a
    wrapper that ignores the disabled keys and forwards any other key
    to matplotlib's standard handler. Other figures are not affected.

    Successive calls accumulate the disabled keys: the keys passed in
    each call are added to those disabled in previous calls, until
    `restore_default_keys` is called.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure created through pyplot (it must have a canvas manager and
        use matplotlib's default key handler).
    keys : str or iterable of str
        Key string, or key strings, to ignore (e.g. 'p', ['p', 'l'] or
        'ctrl+s'). The strings are compared with ``event.key``, so they
        are case sensitive ('l' and 'L' are different keys).

    Returns
    -------
    cid : int
        Connection id of the new key handler.

    Raises
    ------
    ValueError
        If the figure has no canvas manager.
    NotImplementedError
        If the figure uses the ToolManager to handle the keys.
    """
    # a single string is one key, not a sequence of one-character keys
    if isinstance(keys, str):
        keys = [keys]
    manager = _get_manager(fig)

    blocked = set(getattr(fig, _DISABLED_KEYS_ATTR, set())) | set(keys)
    setattr(fig, _DISABLED_KEYS_ATTR, blocked)

    def filtered_handler(event):
        if event.key in blocked:
            return
        key_press_handler(event, fig.canvas)

    fig.canvas.mpl_disconnect(manager.key_press_handler_id)
    cid = fig.canvas.mpl_connect("key_press_event", filtered_handler)
    manager.key_press_handler_id = cid
    return cid


def restore_default_keys(fig):
    """Restore matplotlib's standard key handler for a single figure.

    All the keys disabled with `disable_keys_in_figure` are enabled
    again. It can also be called when no key has been disabled.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure created through pyplot (it must have a canvas manager and
        use matplotlib's default key handler).

    Returns
    -------
    cid : int
        Connection id of the restored standard key handler.

    Raises
    ------
    ValueError
        If the figure has no canvas manager.
    NotImplementedError
        If the figure uses the ToolManager to handle the keys.
    """
    manager = _get_manager(fig)
    fig.canvas.mpl_disconnect(manager.key_press_handler_id)
    # same connection as the one made by matplotlib when creating the figure
    manager.key_press_handler_id = fig.canvas.mpl_connect("key_press_event", key_press_handler)
    if hasattr(fig, _DISABLED_KEYS_ATTR):
        delattr(fig, _DISABLED_KEYS_ATTR)
    return manager.key_press_handler_id
