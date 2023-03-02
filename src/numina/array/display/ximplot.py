#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np

from .matplotlib_qt import set_window_geometry
from .pause_debugplot import pause_debugplot


def ximplot(ycut, title=None, show=True, plot_bbox=(0, 0),
            geometry=(0, 0, 640, 480), tight_layout=True,
            debugplot=None):
    """Auxiliary function to display 1d plot.

    Parameters
    ----------
    ycut : 1d numpy array, float
        Array to be displayed.
    title : string
        Plot title.
    show : bool
        If True, the function shows the displayed image. Otherwise
        plt.show() is expected to be executed outside.
    plot_bbox : tuple (2 integers)
        If tuple is (0,0), the plot is displayed with image
        coordinates (indices corresponding to the numpy array).
        Otherwise, the bounding box of the image is read from this
        tuple, assuming (nc1,nc2). In this case, the coordinates
        indicate pixels.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the window geometry.
    tight_layout : bool
        If True, and show=True, a tight display layout is set.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. This value is returned only when
        'show' is False.

    """

    # protections
    if type(ycut) is not np.ndarray:
        raise ValueError("ycut=" + str(ycut) +
                         " must be a numpy.ndarray")
    elif ycut.ndim != 1:
        raise ValueError("ycut.ndim=" + str(ycut.dim) +
                         " must be 1")

    # read bounding box limits
    nc1, nc2 = plot_bbox
    plot_coord = (nc1 == 0 and nc2 == 0)

    naxis1_ = ycut.size
    if not plot_coord:
        # check that ycut size corresponds to bounding box size
        if naxis1_ != nc2 - nc1 + 1:
            raise ValueError("ycut.size=" + str(ycut.size) +
                             " does not correspond to bounding box size")

    # display image
    from numina.array.display.matplotlib_qt import plt
    if not show:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.autoscale(False)
    ymin = ycut.min()
    ymax = ycut.max()
    if plot_coord:
        xmin = -0.5
        xmax = (naxis1_ - 1) + 0.5
        xcut = np.arange(naxis1_, dtype=float)
        ax.set_xlabel('image array index in the X direction')
        ax.set_ylabel('pixel value')
    else:
        xmin = float(nc1) - 0.5
        xmax = float(nc2) + 0.5
        xcut = np.linspace(start=nc1, stop=nc2, num=nc2 - nc1 + 1)
        ax.set_xlabel('image pixel in the X direction')
        ax.set_ylabel('pixel value')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.plot(xcut, ycut, '-')
    if title is not None:
        ax.set_title(title)

    # set the geometry
    set_window_geometry(geometry)

    if show:
        pause_debugplot(debugplot, pltshow=show, tight_layout=tight_layout)
    else:
        if tight_layout:
            plt.tight_layout()
        # return axes
        return ax
