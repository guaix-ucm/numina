#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""ximplotxy.py: Plot X-Y data from ASCII files or numpy arrays.

Example of usage from command line:
    $ numina-ximplotxy data.txt 1 2 --kwargs "{'marker':'o','linestyle':''}"
    $ numina-ximplotxy lineas.txt 1 2 --vlines --kwargs "{'color':'red', 'xlabel':'wavelength', 'ylabel':'flux'}"


"""

import argparse
import numpy as np

from .matplotlib_qt import set_window_geometry
from .pause_debugplot import pause_debugplot


def ximplotxy_jupyter(x, y, fmt=None, **args):
    """Auxiliary function to call ximplotxy from a jupyter notebook."""
    using_jupyter = True
    if fmt is None:
        return ximplotxy(x, y, using_jupyter=using_jupyter, **args)
    else:
        return ximplotxy(x, y, fmt, using_jupyter=using_jupyter, **args)


def ximplotxy(
    x,
    y,
    fmt=None,
    plottype=None,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    title=None,
    show=True,
    geometry=(0, 0, 640, 480),
    tight_layout=True,
    debugplot=0,
    using_jupyter=False,
    fig=None,
    **kwargs,
):
    """
    Parameters
    ----------
    x : 1d numpy array, float
        Array containing the X coordinate.
    y : 1d numpy array, float
        Array containing the Y coordinate.
    fmt : str, optional
        Format string for quickly setting basic line properties.
    plottype : string
        Plot type. It can be 'loglog', 'semilogx', 'semilogy', 'vlines'
        or None (default, non-logarithmic plot). With 'vlines', a
        vertical line from y=0 to y[i] is drawn at each x[i] (e.g. to
        simulate an emission-line spectrum); in this case 'fmt' is
        ignored and 'kwargs' are passed to matplotlib's Axes.vlines.
        If 'ylim' is None and all y values are non-negative, the lower
        limit of the y-axis is set to zero.
    xlim : tuple of floats
        Tuple defining the x-axis range.
    ylim : tuple of floats
        Tuple defining the y-axis range.
    xlabel : string
        X-axis label.
    ylabel : string
        Y-axis label.
    title : string
        Plot title.
    show : bool
        If True, the function shows the displayed image. Otherwise
        plt.show() is expected to be executed outside.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the window geometry.
    tight_layout : bool
        If True, and show=True, a tight display layout is set.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.
    using_jupyter : bool
        If True, this function is called from a jupyter notebook.
    fig : matplotlib.figure.Figure or None
        Figure where the plot is drawn, for example a figure created
        without pyplot to be saved in a file. The axes are returned,
        and 'show', 'geometry' and 'debugplot' are not used. If None,
        a new figure is created with pyplot.

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. This value is returned only when
        'show' is False or 'fig' is given.

    """

    if fig is not None:
        ax = fig.add_subplot(111)
        _plotxy(ax, x, y, fmt, plottype, xlim, ylim, xlabel, ylabel, title, **kwargs)
        if tight_layout:
            fig.tight_layout()
        return ax

    from numina.array.display.matplotlib_qt import plt

    if not show and using_jupyter:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    _plotxy(ax, x, y, fmt, plottype, xlim, ylim, xlabel, ylabel, title, **kwargs)

    set_window_geometry(geometry)

    if show:
        pause_debugplot(debugplot, pltshow=show, tight_layout=tight_layout)
    else:
        if tight_layout:
            plt.tight_layout()
        # return axes
        if using_jupyter:
            plt.ion()
        return ax


# Line2D properties (besides those starting with 'marker') that are not
# accepted by the LineCollection returned by Axes.vlines
_LINE2D_ONLY_KWARGS = {"ms", "mfc", "mec", "mew", "fillstyle", "drawstyle", "ds"}


def _plotxy(ax, x, y, fmt, plottype, xlim, ylim, xlabel, ylabel, title, **kwargs):
    """Draw the plot of ximplotxy in the axes ax"""
    if plottype == "loglog":
        if fmt is None:
            ax.loglog(x, y, **kwargs)
        else:
            ax.loglog(x, y, fmt, **kwargs)
    elif plottype == "semilogx":
        if fmt is None:
            ax.semilogx(x, y, **kwargs)
        else:
            ax.semilogx(x, y, fmt, **kwargs)
    elif plottype == "semilogy":
        if fmt is None:
            ax.semilogy(x, y, **kwargs)
        else:
            ax.semilogy(x, y, fmt, **kwargs)
    elif plottype == "vlines":
        # vertical lines starting at y=0; Axes.vlines returns a
        # LineCollection, which does not accept a format string nor
        # the marker/drawstyle properties of Line2D
        if fmt is not None:
            raise ValueError("fmt cannot be used with plottype='vlines'")
        invalid_kwargs = [key for key in kwargs if key.startswith("marker") or key in _LINE2D_ONLY_KWARGS]
        if invalid_kwargs:
            raise ValueError(f"Invalid argument(s) for plottype='vlines': {', '.join(invalid_kwargs)}")
        ax.vlines(x, 0, y, **kwargs)
    elif plottype is None:
        if fmt is None:
            ax.plot(x, y, **kwargs)
        else:
            ax.plot(x, y, fmt, **kwargs)
    else:
        raise ValueError("Invalid plottype: " + str(plottype))

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    elif plottype == "vlines" and np.min(y) >= 0:
        # lines start at y=0: remove the autoscale margin below zero
        # (kept when there are negative values, to avoid hiding them)
        ax.set_ylim(bottom=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog="ximplotxy")
    parser.add_argument("filename", help="ASCII file with data in columns")
    parser.add_argument("col1", help="Column number for X data", type=int)
    parser.add_argument("col2", help="Column number for Y data", type=int)
    parser.add_argument(
        "--vlines",
        help="Draw a vertical line from y=0 at each point (emission-line spectrum)",
        action="store_true",
    )
    parser.add_argument(
        "--kwargs",
        help="Extra arguments for plot, e.g.: "
        "\"{'marker':'o',"
        " 'linestyle':'dotted',"
        " 'xlabel':'x axis', 'ylabel':'y axis',"
        " 'title':'sample plot',"
        " 'xlim':[-1,1], 'ylim':[-2,2],"
        " 'label':'sample data',"
        " 'color':'magenta'}\"",
    )
    parser.add_argument("--xlim", help="X-axis limits", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", help="Y-axis limits", type=float, nargs=2, metavar=("YMIN", "YMAX"))
    parser.add_argument("--geometry", help="tuple x,y,dx,dy (default 0,0,640,480)", default="0,0,640,480")
    args = parser.parse_args(args)

    # ASCII file
    filename = args.filename

    # columns to be plotted (first column will be number 1 and not 0)
    col1 = args.col1 - 1
    col2 = args.col2 - 1

    # geometry
    geometry = tuple(int(item) for item in args.geometry.split(","))

    # read ASCII file
    bigtable = np.genfromtxt(filename)
    x = bigtable[:, col1]
    y = bigtable[:, col2]

    plottype = "vlines" if args.vlines else None

    if args.kwargs is None:
        if args.vlines:
            plot_kwargs = {}
        else:
            plot_kwargs = {"marker": "o", "linestyle": ""}
    else:
        plot_kwargs = eval(args.kwargs)

    # axis limits: --xlim/--ylim cannot be combined with the same keys in --kwargs
    for limname in ("xlim", "ylim"):
        limvalue = getattr(args, limname)
        if limvalue is not None:
            if limname in plot_kwargs:
                parser.error(f"--{limname} cannot be used together with '{limname}' in --kwargs")
            plot_kwargs[limname] = limvalue

    ximplotxy(x, y, plottype=plottype, geometry=geometry, debugplot=12, **plot_kwargs)


if __name__ == "__main__":
    main()
