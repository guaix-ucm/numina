#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import argparse
import numpy as np

from .matplotlib_qt import set_window_geometry
from .pause_debugplot import pause_debugplot


def ximplotxy_jupyter(x, y, fmt=None, **args):
    """Auxiliary function to call ximplotxy from a jupyter notebook.
    """
    using_jupyter = True
    if fmt is None:
        return ximplotxy(x, y, using_jupyter=using_jupyter, **args)
    else:
        return ximplotxy(x, y, fmt, using_jupyter=using_jupyter, **args)



def ximplotxy(x, y, fmt=None, plottype=None,
              xlim=None, ylim=None, 
              xlabel=None, ylabel=None, title=None,
              show=True, geometry=(0, 0, 640, 480), tight_layout=True,
              debugplot=0, using_jupyter=False,
              **kwargs):
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
        Plot type. It can be 'loglog', 'semilogx', 'semilogy' or None
        (default, non-logarithmic plot).
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

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. This value is returned only when
        'show' is False.

    """

    from numina.array.display.matplotlib_qt import plt
    if not show and using_jupyter:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if plottype == 'loglog':
        if fmt is None:
            ax.loglog(x, y, **kwargs)
        else:
            ax.loglog(x, y, fmt, **kwargs)
    elif plottype == 'semilogx':
        if fmt is None:
            ax.semilogx(x, y, **kwargs)
        else:
            ax.semilogx(x, y, fmt, **kwargs)
    elif plottype == 'semilogy':
        if fmt is None:
            ax.semilogy(x, y, **kwargs)
        else:
            ax.semilogy(x, y, fmt, **kwargs)
    elif plottype == None:
        if fmt is None:
            ax.plot(x, y, **kwargs)
        else:
            ax.plot(x, y, fmt, **kwargs)
    else:
        raise ValueError('Invalid plottype: ' + str(plottype))

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

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


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='ximplotxy')
    parser.add_argument("filename",
                        help="ASCII file with data in columns")
    parser.add_argument("col1",
                        help="Column number for X data",
                        type=int)
    parser.add_argument("col2",
                        help="Column number for Y data",
                        type=int)
    parser.add_argument("--kwargs",
                        help="Extra arguments for plot, e.g.: "
                             "\"{'marker':'o',"
                             " 'linestyle':'dotted',"
                             " 'xlabel':'x axis', 'ylabel':'y axis',"
                             " 'title':'sample plot',"
                             " 'xlim':[-1,1], 'ylim':[-2,2],"
                             " 'label':'sample data',"
                             " 'color':'magenta'}\"")
    args = parser.parse_args(args)

    # ASCII file
    filename = args.filename

    # columns to be plotted (first column will be number 1 and not 0)
    col1 = args.col1 - 1
    col2 = args.col2 - 1

    # read ASCII file
    bigtable = np.genfromtxt(filename)
    x = bigtable[:, col1]
    y = bigtable[:, col2]

    if args.kwargs is None:
        ximplotxy(x, y, debugplot=12, marker='o', linestyle='')
    else:
        ximplotxy(x, y, debugplot=12, **eval(args.kwargs))


if __name__ == '__main__':
    main()
