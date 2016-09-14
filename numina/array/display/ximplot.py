#
# Copyright 2015-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import division
from __future__ import print_function

import numpy as np

from .pause_debugplot import pause_debugplot


def ximplot(ycut, title=None, show=True, plot_bbox=(0, 0), debugplot=None):
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
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

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
    elif ycut.ndim is not 1:
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
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    # plt.ion()
    # plt.pause(0.001)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.autoscale(False)
    ymin = ycut.min()
    ymax = ycut.max()
    if plot_coord:
        xmin = -0.5
        xmax = (naxis1_ - 1) + 0.5
        xcut = np.arange(naxis1_, dtype=np.float)
        ax.set_xlabel('image array index in the X direction')
        ax.set_ylabel('pixel value')
    else:
        xmin = float(nc1) - 0.5
        xmax = float(nc2) + 0.5
        xcut = np.linspace(start=nc1, stop=nc2, num=nc2 - nc1 + 1)
        ax.set_xlabel('image pixel in the X direction')
        ax.set_ylabel('pixel value')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.plot(xcut, ycut, '-')
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(debugplot)
    else:
        # return axes
        return ax
