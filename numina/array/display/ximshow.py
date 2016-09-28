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

import argparse
from astropy.io import fits
import numpy as np

from .pause_debugplot import pause_debugplot
from .list_fits_files_from_txt import list_fits_files_from_txt
from ..stats import summary

from numina.visualization import ZScaleInterval


dum_str = ''  # global variable in function keypress
dum_par = ''  # global variable in function keypress


def ximshow(image2d, title=None, cbar_label=None, show=True,
            z1z2=None, cmap="hot", image_bbox=(0, 0, 0, 0),
            geometry=None, debugplot=None):
    """Auxiliary function to display a numpy 2d array.

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
    z1z2 : tuple of floats, string or None
        Background and foreground values. If None, zcuts are employed.
    cmap : string
        Color map to be employed.
    image_bbox : tuple (4 integers)
        If (0,0,0,0), image is displayed with image coordinates
        (indices corresponding to the numpy array). Otherwise,
        the bounding box of the image is read from this tuple,
        assuming (nc1,nc2,ns1,ns2). In this case, the coordinates
        indicate pixels.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the Qt4 backend geometry.
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

    if image_coord:
        nc1 = 1
        nc2 = naxis1_
        ns1 = 1
        ns2 = naxis2_

    # if not image_coord:
    #     # check that image shape corresponds to bounding box size
    #     if naxis1_ != nc2 - nc1 + 1:
    #         raise ValueError("image2d.shape=" + str(image2d.shape) +
    #                          " does not correspond to bounding box size")
    #     if naxis2_ != ns2 - ns1 + 1:
    #         raise ValueError("image2d.shape=" + str(image2d.shape) +
    #                          " does not correspond to bounding box size")

    def get_current_zoom(ax_image, debug=False):
        """Return subimage corresponding to current zoom.

        Parameters
        ----------
        ax_image : axes
            Image axes.
        debug : bool
            If True, the image corners are printed.

        Returns
        -------
        subimage : numpy array (floats)
            Subimage.

        """

        xmin_image, xmax_image = ax_image.get_xlim()
        ymin_image, ymax_image = ax_image.get_ylim()
        ixmin = int(xmin_image + 0.5)
        ixmax = int(xmax_image + 0.5)
        iymin = int(ymin_image + 0.5)
        iymax = int(ymax_image + 0.5)
        if ixmin < nc1 - 1:
            ixmin = nc1 - 1
        if ixmin > nc2 - 1:
            ixmin = nc2 - 1
        if ixmax < nc1 - 1:
            ixmax = nc1 - 1
        if ixmax > nc2 - 1:
            ixmax = nc2 - 1
        if iymin < ns1 - 1:
            iymin = ns1 - 1
        if iymin > ns2 - 1:
            iymin = ns2 - 1
        if iymax < ns1 - 1:
            iymax = ns1 - 1
        if iymax > ns2 - 1:
            iymax = ns2 - 1
        if debug:
            print("\n>>> xmin, xmax, ymin, ymax (pixels):",
                  ixmin+1, ixmax+1, iymin+1, iymax+1)
        return image2d[iymin:(iymax+1), ixmin:(ixmax+1)]

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
        if event.key == "?":
            print("""
Keyword events
==============
Home/Reset......................: h or r or home
Back............................: c or left arrow or backspace
Forward.........................: v or right arrow
Pan/Zoom........................: p
Zoom-to-rect....................: o
Save............................: ctrl + s
Toggle fullscreen...............: ctrl + f
Close plot......................: ctrl + w
Set zscale......................: /
Set bg=min and fg=max values....: ,
Display statistical summary.....: .
Set foreground by keyboard......: m
Set background by keyboard......: n
Constrain pan/zoom to x axis....: hold x when panning/zooming with mouse
Constrain pan/zoom to y axis....: hold y when panning/zooming with mouse
Preserve aspect ratio...........: hold CONTROL when panning/zooming with mouse
Toggle grid.....................: g when mouse is over an axes
Toggle x axis scale (log/linear): L or k when mouse is over an axes
Toggle y axis scale (log/linear): l when mouse is over an axes
            """)
        elif event.key == "/":
            subimage2d = get_current_zoom(ax, debug=True)
            new_vmin, new_vmax = ZScaleInterval().get_limits(subimage2d)
            print(">>> setting cuts to vmin=" + str(new_vmin) +
                  " and vmax=" + str(new_vmax))
            im_show.set_clim(vmin=new_vmin)
            im_show.set_clim(vmax=new_vmax)
            dum_str = ''
            dum_par = ''
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == ",":
            subimage2d = get_current_zoom(ax, debug=True)
            new_vmin = subimage2d.min()
            new_vmax = subimage2d.max()
            print(">>> setting cuts to vmin=" + str(new_vmin) +
                  " and vmax=" + str(new_vmax))
            im_show.set_clim(vmin=new_vmin)
            im_show.set_clim(vmax=new_vmax)
            dum_str = ''
            dum_par = ''
            plt.show(block=False)
            plt.pause(0.001)
        elif event.key == ".":
            subimage2d = get_current_zoom(ax, debug=True)
            summary(subimage2d.flatten(), debug=True)
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
    ax.grid(False)
    if z1z2 is None:
        z1, z2 = ZScaleInterval().get_limits(
            image2d[(ns1 - 1):ns2, (nc1 - 1):nc2]
        )
    elif z1z2 == "minmax":
        z1 = image2d[(ns1 - 1):ns2, (nc1 - 1):nc2].min()
        z2 = image2d[(ns1 - 1):ns2, (nc1 - 1):nc2].max()
    else:
        z1, z2 = z1z2
    im_show = plt.imshow(image2d[(ns1 - 1):ns2, (nc1 - 1):nc2],
                         cmap=cmap, aspect='auto',
                         vmin=z1, vmax=z2,
                         interpolation='nearest', origin='low',
                         extent=[xmin, xmax, ymin, ymax])
    if cbar_label is None:
        cbar_label = "Number of counts"
    plt.colorbar(im_show, shrink=1.0, label=cbar_label,
                 orientation="horizontal")
    if title is not None:
        ax.set_title(title)

    # set the geometry
    if geometry is not None:
        x_geom, y_geom, dx_geom, dy_geom = geometry
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(x_geom, y_geom, dx_geom, dy_geom)
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


def ximshow_file(singlefile,
                 args_z1z2=None, args_bbox=None,
                 args_keystitle=None, args_geometry=None,
                 args_pdffile=None,
                 args_debugplot=None):
    """Function to execute ximshow() as called from command line.

    Parameters
    ----------
    singlefile : string
        Name of the FITS file to be displayed.
    args_z1z2 : string or None
        String providing the image cuts tuple: z1, z2, minmax of None
    args_bbox : string or None
        String providing the bounding box tuple: nc1, nc2, ns1, ns2
    args_keystitle : string or None
        Tuple of FITS keywords.format: key1,key2,...,keyn.format
    args_geometry : string or None
        Tuple x,y,dx,dy to define the Qt4 backend geometry. This
        information is ignored if args_pdffile is not None.
    args_pdffile : string or None
        Output PDF file name.
    args_debugplot : string or None
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    """

    # read z1, z2
    if args_z1z2 is None:
        z1z2 = None
    elif args_z1z2 == "minmax":
        z1z2 = "minmax"
    else:
        tmp_str = args_z1z2.split(",")
        z1z2 = float(tmp_str[0]), float(tmp_str[1])

    # read pdffile
    pdffile = args_pdffile
    if pdffile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(pdffile)
    else:
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        pdf = None

    # read geometry
    if args_geometry is None:
        geometry = None
    else:
        tmp_str = args_geometry.split(",")
        x_geom = int(tmp_str[0])
        y_geom = int(tmp_str[1])
        dx_geom = int(tmp_str[2])
        dy_geom = int(tmp_str[3])
        geometry = x_geom, y_geom, dx_geom, dy_geom

    # read debugplot value
    debugplot = int(args_debugplot)

    # read input FITS file
    hdulist = fits.open(singlefile)
    image_header = hdulist[0].header
    image2d = hdulist[0].data
    hdulist.close()

    naxis1 = image_header['naxis1']
    naxis2 = image_header['naxis2']

    # title for plot
    title = singlefile
    if args_keystitle is not None:
        keystitle = args_keystitle
        keysformat = ".".join(keystitle.split(".")[1:])
        keysnames = keystitle.split(".")[0]
        tuple_of_keyval = ()
        for key in keysnames.split(","):
            keyval = image_header[key]
            tuple_of_keyval += (keyval,)
        title += "\n" + str(keysformat % tuple_of_keyval)

    if image2d.shape != (naxis2, naxis1):
        raise ValueError("Unexpected error with NAXIS1, NAXIS2")
    else:
        print('>>> File..:', singlefile)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    # read bounding box
    if args_bbox is None:
        nc1 = 1
        nc2 = naxis1
        ns1 = 1
        ns2 = naxis2
    else:
        tmp_bbox = args_bbox.split(",")
        nc1 = int(tmp_bbox[0])
        nc2 = int(tmp_bbox[1])
        ns1 = int(tmp_bbox[2])
        ns2 = int(tmp_bbox[3])
        if nc1 < 1:
            nc1 = 1
        if nc2 > naxis1:
            nc2 = naxis1
        if ns1 < 1:
            ns1 = 1
        if ns2 > naxis2:
            ns2 = naxis2

    # display image
    ax = ximshow(image2d=image2d, show=False,
                 title=title,
                 z1z2=z1z2,
                 image_bbox=(nc1, nc2, ns1, ns2),
                 geometry=geometry,
                 debugplot=debugplot)

    if pdf is not None:
        pdf.savefig()
    else:
        import matplotlib.pyplot as plt
        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(debugplot)

    if pdf is not None:
        pdf.close()


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='ximshow')
    parser.add_argument("filename",
                        help="FITS file or txt file with list of FITS files")
    parser.add_argument("--z1z2",
                        help="tuple z1,z2, minmax or None (use zscale)")
    parser.add_argument("--bbox",
                        help="bounding box tuple: nc1,nc2,ns1,ns2")
    parser.add_argument("--keystitle",
                        help="tuple of FITS keywords.format: " +
                             "key1,key2,...keyn.'format'")
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy")
    parser.add_argument("--pdffile",
                        help="ouput PDF file name")
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=12)",
                        default=12)
    args = parser.parse_args(args)

    list_fits_files = list_fits_files_from_txt(args.filename)

    for myfile in list_fits_files:
        ximshow_file(singlefile=myfile,
                     args_z1z2=args.z1z2,
                     args_bbox=args.bbox,
                     args_keystitle=args.keystitle,
                     args_geometry=args.geometry,
                     args_pdffile=args.pdffile,
                     args_debugplot=args.debugplot)

    if len(list_fits_files) > 1:
        pause_debugplot(12, optional_prompt="Press RETURN to STOP")


if __name__ == "__main__":

    main()
