#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import logging

_logger = logging.getLogger(__name__)

# the warning about the window position is only shown once
_warned_position = False


def _pyplot():
    """Import matplotlib.pyplot, configured, the first time it is needed

    Importing pyplot takes some tenths of a second, and the modules that
    import this one only plot in some functions.
    """
    import matplotlib.pyplot as plt

    if "plt" not in globals():
        plt.rcParams.update({"figure.max_open_warning": 0})  # avoid warning
        globals()["plt"] = plt
    return plt


def __getattr__(name):
    # plt is imported on first access
    if name == "plt":
        return _pyplot()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def new_figure(interactive=True, **kwargs):
    """Create a figure, managed by pyplot only if it is displayed

    A figure that is only saved to a file does not need pyplot: it is
    independent of the backend and of the other figures, and it is
    freed when it is no longer used.

    Parameters
    ==========
    interactive : bool
        If True, the figure is created with pyplot, to be displayed.
    **kwargs
        Arguments of the Figure.

    """
    if interactive:
        return _pyplot().figure(**kwargs)

    from matplotlib.figure import Figure

    return Figure(**kwargs)


def set_window_geometry(geometry):
    """Set window geometry.

    With the Qt backends both the position and the size of the window
    are set; with the Tk backend too. The macosx backend does not allow
    to set the window position, so only the figure size is set.

    Parameters
    ==========
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the window geometry.

    """

    global _warned_position

    if geometry is None:
        return

    x_geom, y_geom, dx_geom, dy_geom = geometry
    plt = _pyplot()
    mngr = plt.get_current_fig_manager()
    if mngr is None:
        return
    window = getattr(mngr, "window", None)
    if hasattr(window, "setGeometry"):
        # Qt
        window.setGeometry(x_geom, y_geom, dx_geom, dy_geom)
    elif hasattr(window, "wm_geometry"):
        # Tk
        window.wm_geometry(f"{dx_geom}x{dy_geom}+{x_geom}+{y_geom}")
    else:
        fig = mngr.canvas.figure
        fig.set_size_inches(dx_geom / 640 * 6.4, dy_geom / 480 * 4.8)
        # show warning only once and only if log-level is WARNING or lower
        if _logger.isEnabledFor(logging.WARNING):
            _logger.warning(
                "window position cannot be set with backend %s (only its size); try MPLBACKEND=QtAgg",
                plt.get_backend(),
            )
            _warned_position = True
