#
# Copyright 2015 Universidad Complutense de Madrid
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
from .zscale import zscale


dum_str = ''  # global variable in function keypress
dum_par = ''  # global variable in function keypress


def ximshow(image2d, title=None, cbar_label=None, show=True,
            z1z2=None, cmap="hot", image_bbox=(0, 0, 0, 0),
            debugplot=None):
    """Auxiliary function to display 2d images.

    Parameters
    ----------
    image2d : 2d numpy array, float
        2d image to be displayed.
    title : string
        Plot title.
    cbar_label : string
        Color bar label.
    show : bool
        If True, the function shows the displayed image. Otherwise
        the function just invoke the plt.imshow() function and
        plt.show() is expected to be executed outside.
    z1z2 : tuple of floats
        Background and foreground values. If None, zcuts are employed.
    cmap : string
        Color map to be employed.
    image_bbox : tuple (4 integers)
        If (0,0,0,0), image is displayed with image coordinates
        (indices corresponding to the numpy array). Otherwise,
        the bounding box of the image is read from this tuple,
        assuming (nc1,nc2,ns1,ns2). In this case, the coordinates
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
        This parameter is ignored when show is False.

    Returns
    -------
    ax : axes object
        Matplotlib axes instance. This value is returned only when
        'show' is False.

    """

    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt

    # protections
    if type(image2d) is not np.ndarray:
        raise ValueError("image2d=" + str(image2d) +
                         " must be a numpy.ndarray")
    elif image2d.ndim is not 2:
        raise ValueError("image2d.ndim=" + str(image2d.dim) +
                         " must be 2")

    # read bounding box limits
    nc1, nc2, ns1, ns2 = image_bbox
    image_coord = (nc1 == 0 and nc2 == 0 and ns1 == 0 and ns2 == 0)

    naxis2_, naxis1_ = image2d.shape
    if not image_coord:
        # check that image shape corresponds to bounding box size
        if naxis1_ != nc2 - nc1 + 1:
            raise ValueError("image2d.shape=" + str(image2d.shape) +
                             " does not correspond to bounding box size")
        if naxis2_ != ns2 - ns1 + 1:
            raise ValueError("image2d.shape=" + str(image2d.shape) +
                             " does not correspond to bounding box size")

    def keypress(event):
        """Deal with keyboard events, allowing the update of vmin and vmax.

        Note that a call to raw_input() is not allowed within this
        function since, in that case, the following runtime error
        is raised: can't re-enter readline

        For that reason, the new vmin and vmax values should be
        entered blindly.

        To avoid collisions with navigation keyboard shortcuts,
        check the table available at:
        http://matplotlib.org/users/navigation_toolbar.html

        """

        global dum_str
        global dum_par
        if event.key == "/":
            new_vmin, new_vmax = zscale(image2d)
            print("Setting cuts to vmin=" + str(new_vmin) +
                  " and vmax=" + str(new_vmax))
            im_show.set_clim(vmin=new_vmin)
            im_show.set_clim(vmax=new_vmax)
            dum_str = ''
            dum_par = ''
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == ",":
            new_vmin = image2d.min()
            new_vmax = image2d.max()
            print("Setting cuts to vmin=" + str(new_vmin) +
                  " and vmax=" + str(new_vmax))
            im_show.set_clim(vmin=new_vmin)
            im_show.set_clim(vmax=new_vmax)
            dum_str = ''
            dum_par = ''
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == "n":
            print("Type (blindly!) vmin <return>")
            dum_str = ''
            dum_par = "vmin"
        elif event.key == "m":
            print("Type (blindly!) vmax <return>")
            dum_str = ''
            dum_par = "vmax"
        elif event.key == "enter":
            if dum_par == "vmin":
                try:
                    new_vmin = float(dum_str)
                except ValueError:
                    print("Invalid vmin=" + dum_str)
                    dum_str = ''
                    print("Type again (blindly!) vmin <return>")
                else:
                    print("Setting vmin=" + dum_str)
                    im_show.set_clim(vmin=new_vmin)
                    dum_str = ''
                    dum_par = ''
                    plt.show(block=False)
                    plt.pause(0.001)
            elif dum_par == "vmax":
                try:
                    new_vmax = float(dum_str)
                except ValueError:
                    print("Invalid vmax=" + dum_str)
                    dum_str = ''
                    print("Type again (blindly!) vmax <return>")
                else:
                    print("Setting vmax=" + dum_str)
                    im_show.set_clim(vmax=new_vmax)
                    dum_str = ''
                    dum_par = ''
                    plt.show(block=False)
                    plt.pause(0.001)
        else:
            if dum_str == '':
                dum_str = event.key
            else:
                dum_str += event.key

    # display image
    # plt.ion()
    # plt.pause(0.001)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.autoscale(False)
    if image_coord:
        xmin = -0.5
        xmax = (naxis1_ - 1) + 0.5
        ymin = -0.5
        ymax = (naxis2_ - 1) + 0.5
        ax.set_xlabel('image array index in the X direction')
        ax.set_ylabel('image array index in the Y direction')
    else:
        xmin = float(nc1) - 0.5
        xmax = float(nc2) + 0.5
        ymin = float(ns1) - 0.5
        ymax = float(ns2) + 0.5
        ax.set_xlabel('image pixel in the X direction')
        ax.set_ylabel('image pixel in the Y direction')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if z1z2 is None:
        z1, z2 = zscale(image2d)
    else:
        z1, z2 = z1z2
    im_show = plt.imshow(image2d, cmap=cmap, aspect='auto',
                         vmin=z1, vmax=z2,
                         interpolation='nearest', origin='low',
                         extent=[xmin, xmax, ymin, ymax])
    if cbar_label is None:
        cbar_label = "Number of counts"
    plt.colorbar(im_show, shrink=1.0, label=cbar_label,
                 orientation="horizontal")
    if title is not None:
        ax.set_title(title)
    # connect keypress event with function responsible for
    # updating vmin and vmax
    fig.canvas.mpl_connect('key_press_event', keypress)
    # show plot or return axes
    if show:
        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(debugplot)
    else:
        # return axes
        return ax
